import pandas as pd
import numpy as np
import xgboost as xgb
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix

def run_simplified_champion():
    print("Loading core dataset...")
    df = pd.read_csv("for_participants/data/players_quarters_final.csv")
    
    # 1. Feature Engineering
    abs_min_map = {'half_1': 0, 'half_2': 45, 'extra_time_1': 90}
    df['checkpoint_abs_min'] = df['checkpoint_period'].map(abs_min_map) + df['checkpoint_min']
    df['mins_on_pitch'] = df['checkpoint_abs_min'] - df['minute_in']
    
    team_group = df.groupby(['fixture_id', 'is_home', 'checkpoint'])
    df['team_last15_shots'] = team_group['last15_shots'].transform('sum')
    df['team_last15_sprints'] = team_group['last15_sprints'].transform('sum')
    df['team_last15_distance'] = team_group['last15_distance'].transform('sum')
    
    # 2. Position Aggregation (Aggregating A and M into a single threat variable)
    df['is_attacker_mid'] = df['position'].isin(['A', 'M']).astype(int)
    
    target = 'scored_after'
    groups = df['fixture_id']
    y = df[target]
    
    # 3. Feature Selection (Removing Formation and Position, adding the new binary)
    drop_cols = [
        'player_appearance_id', 'player_id', 'fixture_id', 'date', 
        'checkpoint', 'checkpoint_period', 'checkpoint_min',
        'minute_out', 'subbed', 'jersey_number', 'checkpoint_abs_min', 'minute_in',
        'formation', 'position'  # Specifically requested removals
    ]
    
    X = df.drop(columns=drop_cols + [target])
    
    # Split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # 4. Champion Parameters
    best_params = {
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
    
    print(f"Training Simplified Model with {X.shape[1]} features...")
    base_model = xgb.XGBClassifier(**best_params)
    simplified_model = BalancedBaggingClassifier(
        estimator=base_model,
        n_estimators=15,
        sampling_strategy='not minority',
        replacement=False,
        random_state=42
    )
    simplified_model.fit(X_train, y_train)
    
    # 5. Evaluation
    y_probs = simplified_model.predict_proba(X_test)[:, 1]
    y_pred = simplified_model.predict(X_test)
    
    roc_auc = roc_auc_score(y_test, y_probs)
    pr_auc = average_precision_score(y_test, y_probs)
    
    print("\n--- SIMPLIFIED MODEL PERFORMANCE ---")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Comparison
    print("\n--- Comparison to Previous Champion ---")
    print(f"Previous Champion ROC-AUC: 0.7069")
    print(f"Simplified Model ROC-AUC:  {roc_auc:.4f} (Delta: {roc_auc - 0.7069:+.4f})")
    
    # Feature Importance check
    importances = np.mean([est.steps[-1][1].feature_importances_ for est in simplified_model.estimators_], axis=0)
    feat_imp = pd.DataFrame({'feature': X.columns, 'importance': importances}).sort_values('importance', ascending=False)
    print("\nTop 5 Features in Simplified Model:")
    print(feat_imp.head(5))

if __name__ == "__main__":
    run_simplified_champion()
