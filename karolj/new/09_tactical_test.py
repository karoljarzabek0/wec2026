import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score

def run_tactical_test():
    print("Loading tactical dataset...")
    df = pd.read_csv("karolj/new/players_conservative_tactical.csv")
    
    # 1. Feature Engineering
    abs_min_map = {'half_1': 0, 'half_2': 45, 'extra_time_1': 90}
    df['checkpoint_abs_min'] = df['checkpoint_period'].map(abs_min_map) + df['checkpoint_min']
    df['mins_on_pitch'] = df['checkpoint_abs_min'] - df['minute_in']
    
    # Team Momentum (Basic)
    team_group = df.groupby(['fixture_id', 'is_home', 'checkpoint'])
    df['team_last15_shots'] = team_group['last15_shots'].transform('sum')
    
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
    
    # Model 1: Single Tuned XGBoost
    print("\nTraining Single Tuned XGBoost...")
    tuned_params = {
        'learning_rate': 0.10129,
        'max_depth': 8,
        'n_estimators': 149,
        'scale_pos_weight': (len(y_train)-sum(y_train))/sum(y_train),
        'random_state': 42
    }
    m1 = xgb.XGBClassifier(**tuned_params)
    m1.fit(X_train, y_train)
    print(f"ROC-AUC: {roc_auc_score(y_test, m1.predict_proba(X_test)[:, 1]):.4f}")
    
    # Model 2: Tuned Balanced Bagging
    print("\nTraining Tuned Balanced Bagging...")
    m2 = BalancedBaggingClassifier(
        estimator=xgb.XGBClassifier(**{k:v for k,v in tuned_params.items() if k != 'scale_pos_weight'}),
        n_estimators=15,
        random_state=42
    )
    m2.fit(X_train, y_train)
    print(f"ROC-AUC: {roc_auc_score(y_test, m2.predict_proba(X_test)[:, 1]):.4f}")
    
    # SHAP Analysis
    print("\nSHAP Importance for Tactical Model:")
    explainer = shap.TreeExplainer(m1)
    shap_values = explainer.shap_values(X_test)
    
    imp = pd.DataFrame({
        'feature': X.columns,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)
    
    print(imp[imp['feature'].str.contains('l15_top|l15_press|cumul_explosivity')])
    print("\nTop 10 Global:")
    print(imp.head(10))

if __name__ == "__main__":
    run_tactical_test()
