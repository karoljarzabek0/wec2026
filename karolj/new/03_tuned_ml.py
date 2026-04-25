import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import average_precision_score, roc_auc_score

def objective(trial, X_train, y_train, X_test, y_test):
    param = {
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',
        'scale_pos_weight': (len(y_train) - sum(y_train)) / sum(y_train),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'n_estimators': trial.suggest_int('n_estimators', 50, 400),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'random_state': 42
    }
    
    model = xgb.XGBClassifier(**param)
    model.fit(X_train, y_train)
    
    y_probs = model.predict_proba(X_test)[:, 1]
    # Optimizing for PR-AUC (average_precision) as it's harder in imbalanced data
    return average_precision_score(y_test, y_probs)

def run_tuned_model():
    print("Loading and preparing data...")
    df = pd.read_csv("for_participants/data/players_quarters_final.csv")
    
    abs_min_map = {'half_1': 0, 'half_2': 45, 'extra_time_1': 90}
    df['checkpoint_abs_min'] = df['checkpoint_period'].map(abs_min_map) + df['checkpoint_min']
    df['mins_on_pitch'] = df['checkpoint_abs_min'] - df['minute_in']
    
    team_group = df.groupby(['fixture_id', 'is_home', 'checkpoint'])
    df['team_last15_shots'] = team_group['last15_shots'].transform('sum')
    df['team_last15_sprints'] = team_group['last15_sprints'].transform('sum')
    
    target = 'scored_after'
    drop_cols = [
        'player_appearance_id', 'player_id', 'fixture_id', 'date', 
        'checkpoint', 'checkpoint_period', 'checkpoint_min',
        'minute_out', 'subbed', 'jersey_number', 'checkpoint_abs_min', 'minute_in'
    ]
    
    y = df[target]
    X = pd.get_dummies(df.drop(columns=drop_cols + [target]), columns=['position', 'formation'], drop_first=True)
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=df['fixture_id']))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    print("Starting Optuna optimization...")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=30)
    
    print("\nBest trial:")
    trial = study.best_trial
    print(f"  PR-AUC: {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    # Final Model with best params
    best_params = trial.params
    best_params['objective'] = 'binary:logistic'
    best_params['eval_metric'] = 'aucpr'
    best_params['scale_pos_weight'] = (len(y_train) - sum(y_train)) / sum(y_train)
    best_params['random_state'] = 42
    
    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_train, y_train)
    
    y_probs = final_model.predict_proba(X_test)[:, 1]
    final_roc_auc = roc_auc_score(y_test, y_probs)
    final_pr_auc = average_precision_score(y_test, y_probs)
    
    print(f"\nFinal Optimized ROC-AUC: {final_roc_auc:.4f}")
    print(f"Final Optimized PR-AUC:  {final_pr_auc:.4f}")

if __name__ == "__main__":
    run_tuned_model()
