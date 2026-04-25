import pandas as pd
import numpy as np
import xgboost as xgb
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix

def run_master_model():
    print("Loading tactical dataset...")
    df = pd.read_csv("karolj/new/players_conservative_tactical.csv")
    
    # 1. Feature Engineering (Complete Set)
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
    
    # Drop leaky/proxy
    drop_cols = [
        'player_appearance_id', 'player_id', 'fixture_id', 'date', 
        'checkpoint', 'checkpoint_period', 'checkpoint_min',
        'minute_out', 'subbed', 'jersey_number', 'checkpoint_abs_min', 'minute_in', 'period_val'
    ]
    
    X = pd.get_dummies(df.drop(columns=drop_cols + [target]), columns=['position', 'formation'], drop_first=True)
    
    # Split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # 2. FULL OPTIMIZED PARAMETERS (No shortcuts this time)
    full_best_params = {
        'learning_rate': 0.10129919477568608,
        'max_depth': 8,
        'n_estimators': 149,
        'subsample': 0.7807733062526079,
        'colsample_bytree': 0.7754409018439148,
        'gamma': 0.003619980766962292,
        'min_child_weight': 10,
        'random_state': 42,
        'eval_metric': 'aucpr'
    }
    
    print("Training Master Tuned Bagging Model...")
    base_model = xgb.XGBClassifier(**full_best_params)
    
    master_model = BalancedBaggingClassifier(
        estimator=base_model,
        n_estimators=15,
        sampling_strategy='not minority',
        replacement=False,
        random_state=42
    )
    
    master_model.fit(X_train, y_train)
    
    # 3. Evaluation
    y_probs = master_model.predict_proba(X_test)[:, 1]
    y_pred = master_model.predict(X_test)
    
    roc_auc = roc_auc_score(y_test, y_probs)
    pr_auc = average_precision_score(y_test, y_probs)
    
    print("\n--- MASTER MODEL PERFORMANCE ---")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Compare to "Best of turn 05" (using hardcoded reference)
    print("\n--- Improvement Analysis ---")
    print(f"Previous Best ROC-AUC: 0.7069")
    print(f"New Master ROC-AUC:     {roc_auc:.4f} (Delta: {roc_auc - 0.7069:+.4f})")

if __name__ == "__main__":
    run_master_model()
