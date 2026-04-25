import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

def run_experiment_matrix():
    print("Loading data...")
    df = pd.read_csv("for_participants/data/players_quarters_final.csv")
    
    # 1. Feature Engineering (Shared)
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

    # --- Feature Set 1: Full ---
    X_full = pd.get_dummies(df.drop(columns=common_drops + [target]), columns=['position', 'formation'], drop_first=True)

    # --- Feature Set 2: Simplified ---
    df_s = df.copy()
    df_s['is_attacker_mid'] = df_s['position'].isin(['A', 'M']).astype(int)
    X_simp = df_s.drop(columns=common_drops + ['position', 'formation', target])
    
    # Shared Split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, y, groups=groups))
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Models to test
    models = {
        'RF': RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
        'GBM': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
        'LGBM': lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=8, random_state=42, verbose=-1),
        'CatB': CatBoostClassifier(n_estimators=100, learning_rate=0.1, depth=8, random_state=42, verbose=0, allow_writing_files=False),
        'XGB': xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=8, random_state=42, eval_metric='aucpr')
    }

    results = []

    def evaluate(X_set, y_train, y_test, model_obj, feature_name, correction_name):
        X_tr, X_te = X_set.iloc[train_idx], X_set.iloc[test_idx]
        
        print(f"  Testing {feature_name} | {type(model_obj).__name__} | {correction_name}...")
        
        # Train
        model_obj.fit(X_tr, y_train)
        
        # Eval
        y_probs = model_obj.predict_proba(X_te)[:, 1]
        y_pred = model_obj.predict(X_te)
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        return {
            "Features": feature_name,
            "Model": type(model_obj).__name__.replace('Classifier', '').replace('BalancedBagging', 'BB_'),
            "Correction": correction_name,
            "ROC-AUC": roc_auc_score(y_test, y_probs),
            "PR-AUC": average_precision_score(y_test, y_probs),
            "Prec": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn
        }

    for f_name, X_set in [("Full", X_full), ("Simp", X_simp)]:
        for m_name, m_inst in models.items():
            # Method 1: Balanced Bagging
            bb_model = BalancedBaggingClassifier(estimator=m_inst, n_estimators=10, random_state=42)
            results.append(evaluate(X_set, y_train, y_test, bb_model, f_name, "Balanced Bagging"))
            
            # Method 2: SMOTE-Tomek
            st_pipe = Pipeline([
                ('smt', SMOTETomek(random_state=42)),
                ('clf', m_inst)
            ])
            results.append(evaluate(X_set, y_train, y_test, st_pipe, f_name, "SMOTE-Tomek"))

    res_df = pd.DataFrame(results)
    print("\n--- COMPREHENSIVE EXPERIMENT RESULTS ---")
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(res_df.to_string(index=False))
    
    res_df.to_csv("karolj/new/comprehensive_experiment_results.csv", index=False)

if __name__ == "__main__":
    run_experiment_matrix()
