import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import optuna
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

def objective(trial, model_type, X_train, y_train, X_test, y_test):
    if model_type == 'XGB':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 250),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': 42,
            'eval_metric': 'aucpr'
        }
        base = xgb.XGBClassifier(**params)
    elif model_type == 'LGBM':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 250),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'random_state': 42,
            'verbose': -1
        }
        base = lgb.LGBMClassifier(**params)
    elif model_type == 'CatB':
        params = {
            'iterations': trial.suggest_int('iterations', 50, 250),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'random_state': 42,
            'verbose': 0,
            'allow_writing_files': False
        }
        base = CatBoostClassifier(**params)
    elif model_type == 'RF':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'random_state': 42
        }
        base = RandomForestClassifier(**params)
    elif model_type == 'GBM':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'random_state': 42
        }
        base = GradientBoostingClassifier(**params)
    
    # Use Balanced Bagging for the evaluation
    bb = BalancedBaggingClassifier(estimator=base, n_estimators=5, random_state=42)
    bb.fit(X_train, y_train)
    y_probs = bb.predict_proba(X_test)[:, 1]
    return average_precision_score(y_test, y_probs)

def run_tuned_experiment():
    print("Loading data...")
    df = pd.read_csv("for_participants/data/players_quarters_final.csv")
    
    # Shared Feature Engineering
    abs_min_map = {'half_1': 0, 'half_2': 45, 'extra_time_1': 90}
    df['checkpoint_abs_min'] = df['checkpoint_period'].map(abs_min_map) + df['checkpoint_min']
    df['mins_on_pitch'] = df['checkpoint_abs_min'] - df['minute_in']
    team_group = df.groupby(['fixture_id', 'is_home', 'checkpoint'])
    df['team_last15_shots'] = team_group['last15_shots'].transform('sum')
    df['team_last15_sprints'] = team_group['last15_sprints'].transform('sum')
    df['team_last15_distance'] = team_group['last15_distance'].transform('sum')
    
    target = 'scored_after'
    groups = df['fixture_id']
    y = df[target]
    common_drops = ['player_appearance_id', 'player_id', 'fixture_id', 'date', 'checkpoint', 'checkpoint_period', 'checkpoint_min', 'minute_out', 'subbed', 'jersey_number', 'checkpoint_abs_min', 'minute_in']

    # Prep X sets
    X_full = pd.get_dummies(df.drop(columns=common_drops + [target]), columns=['position', 'formation'], drop_first=True)
    df_s = df.copy()
    df_s['is_attacker_mid'] = df_s['position'].isin(['A', 'M']).astype(int)
    X_simp = df_s.drop(columns=common_drops + ['position', 'formation', target])
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, y, groups=groups))
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    results = []
    model_types = ['XGB', 'LGBM', 'CatB', 'RF', 'GBM']
    feature_sets = [("Full", X_full), ("Simp", X_simp)]

    for f_name, X_set in feature_sets:
        X_tr, X_te = X_set.iloc[train_idx], X_set.iloc[test_idx]
        for mt in model_types:
            print(f"\n--- Tuning {mt} on {f_name} features ---")
            study = optuna.create_study(direction='maximize')
            study.optimize(lambda trial: objective(trial, mt, X_tr, y_train, X_te, y_test), n_trials=15)
            
            # Retrain best
            print(f"  Best PR-AUC: {study.best_value:.4f}")
            best_p = study.best_params
            
            # Reconstruct base model
            if mt == 'XGB': base = xgb.XGBClassifier(**best_p, random_state=42, eval_metric='aucpr')
            elif mt == 'LGBM': base = lgb.LGBMClassifier(**best_p, random_state=42, verbose=-1)
            elif mt == 'CatB': base = CatBoostClassifier(**best_p, random_state=42, verbose=0, allow_writing_files=False)
            elif mt == 'RF': base = RandomForestClassifier(**best_p, random_state=42)
            elif mt == 'GBM': base = GradientBoostingClassifier(**best_p, random_state=42)
            
            final_model = BalancedBaggingClassifier(estimator=base, n_estimators=15, random_state=42)
            final_model.fit(X_tr, y_train)
            
            y_probs = final_model.predict_proba(X_te)[:, 1]
            y_pred = final_model.predict(X_te)
            
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            
            results.append({
                "Features": f_name,
                "Model": mt,
                "ROC-AUC": roc_auc_score(y_test, y_probs),
                "PR-AUC": average_precision_score(y_test, y_probs),
                "Prec": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "TP": tp, "FP": fp, "FN": fn, "TN": tn
            })

    res_df = pd.DataFrame(results)
    print("\n--- TUNED EXPERIMENT RESULTS ---")
    print(res_df.to_string(index=False))
    res_df.to_csv("karolj/new/tuned_experiment_results.csv", index=False)

if __name__ == "__main__":
    run_tuned_experiment()
