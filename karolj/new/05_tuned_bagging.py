import pandas as pd
import numpy as np
import xgboost as xgb
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

def run_tuned_bagging():
    print("Loading core dataset...")
    df = pd.read_csv("for_participants/data/players_quarters_final.csv")
    
    # Feature Engineering (Same as Clean Model)
    abs_min_map = {'half_1': 0, 'half_2': 45, 'extra_time_1': 90}
    df['checkpoint_abs_min'] = df['checkpoint_period'].map(abs_min_map) + df['checkpoint_min']
    df['mins_on_pitch'] = df['checkpoint_abs_min'] - df['minute_in']
    
    team_group = df.groupby(['fixture_id', 'is_home', 'checkpoint'])
    df['team_last15_shots'] = team_group['last15_shots'].transform('sum')
    df['team_last15_sprints'] = team_group['last15_sprints'].transform('sum')
    df['team_last15_distance'] = team_group['last15_distance'].transform('sum')
    
    target = 'scored_after'
    drop_cols = [
        'player_appearance_id', 'player_id', 'fixture_id', 'date', 
        'checkpoint', 'checkpoint_period', 'checkpoint_min',
        'minute_out', 'subbed', 'jersey_number', 'checkpoint_abs_min', 'minute_in'
    ]
    
    y = df[target]
    X_raw = df.drop(columns=drop_cols + [target])
    X = pd.get_dummies(X_raw, columns=['position', 'formation'], drop_first=True)
    
    # Grouped Split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=df['fixture_id']))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Optimized Parameters from Trial 2 of Optuna
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
    
    print("Initializing Base Tuned XGBoost...")
    base_model = xgb.XGBClassifier(**best_params)
    
    print("Training Tuned BalancedBagging (15 bags)...")
    bagging_model = BalancedBaggingClassifier(
        estimator=base_model,
        n_estimators=15,
        sampling_strategy='not minority',
        replacement=False,
        random_state=42
    )
    
    bagging_model.fit(X_train, y_train)
    
    # Evaluation
    y_probs = bagging_model.predict_proba(X_test)[:, 1]
    y_pred = bagging_model.predict(X_test)
    
    roc_auc = roc_auc_score(y_test, y_probs)
    pr_auc = average_precision_score(y_test, y_probs)
    
    print("\n--- Tuned Balanced Bagging Performance ---")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature Importance
    importances = np.mean([est.steps[-1][1].feature_importances_ for est in bagging_model.estimators_], axis=0)
    feat_imp = pd.DataFrame({
        'feature': X.columns,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Features (Tuned Bagging):")
    print(feat_imp.head(10))

if __name__ == "__main__":
    run_tuned_bagging()
