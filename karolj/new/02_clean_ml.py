import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score

def run_clean_model():
    print("Loading core dataset...")
    df = pd.read_csv("for_participants/data/players_quarters_final.csv")
    
    # 1. Feature Engineering (Strictly non-leaky)
    print("Engineering non-leaky features...")
    
    # Map checkpoint to absolute minutes
    abs_min_map = {'half_1': 0, 'half_2': 45, 'extra_time_1': 90}
    df['checkpoint_abs_min'] = df['checkpoint_period'].map(abs_min_map) + df['checkpoint_min']
    
    # Minutes on pitch at checkpoint
    df['mins_on_pitch'] = df['checkpoint_abs_min'] - df['minute_in']
    
    # Team-level aggregates (Still from core data)
    team_group = df.groupby(['fixture_id', 'is_home', 'checkpoint'])
    df['team_last15_shots'] = team_group['last15_shots'].transform('sum')
    df['team_last15_sprints'] = team_group['last15_sprints'].transform('sum')
    df['team_last15_distance'] = team_group['last15_distance'].transform('sum')
    
    # Target
    target = 'scored_after'
    
    # Drop identifiers and LEAKY features (minute_out, subbed)
    # Also drop jersey_number to force model to learn from physicals
    drop_cols = [
        'player_appearance_id', 'player_id', 'fixture_id', 'date', 
        'checkpoint', 'checkpoint_period', 'checkpoint_min',
        'minute_out', 'subbed', 'jersey_number', 'checkpoint_abs_min', 'minute_in'
    ]
    
    y = df[target]
    X_raw = df.drop(columns=drop_cols + [target])
    
    # Dummies
    X = pd.get_dummies(X_raw, columns=['position', 'formation'], drop_first=True)
    
    # Grouped Split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=df['fixture_id']))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Train
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='aucpr'
    )
    model.fit(X_train, y_train)
    
    # Eval
    y_probs = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_probs)
    pr_auc = average_precision_score(y_test, y_probs)
    
    print("\n--- Clean Model Performance (No Leakage) ---")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")
    
    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Features (Clean):")
    print(feature_importance.head(10))

if __name__ == "__main__":
    run_clean_model()
