import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.model_selection import GroupShuffleSplit

def run_champion_shap():
    print("Loading core dataset...")
    df = pd.read_csv("for_participants/data/players_quarters_final.csv")
    
    # 1. Feature Engineering (The Champion Feature Set)
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
    
    drop_cols = [
        'player_appearance_id', 'player_id', 'fixture_id', 'date', 
        'checkpoint', 'checkpoint_period', 'checkpoint_min',
        'minute_out', 'subbed', 'jersey_number', 'checkpoint_abs_min', 'minute_in'
    ]
    
    X = pd.get_dummies(df.drop(columns=drop_cols + [target]), columns=['position', 'formation'], drop_first=True)
    
    # Split (Standard 80/20 grouped)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # 2. Champion Parameters
    best_params = {
        'learning_rate': 0.10129919477568608,
        'max_depth': 8,
        'n_estimators': 149,
        'subsample': 0.7807733062526079,
        'colsample_bytree': 0.7754409018439148,
        'gamma': 0.003619980766962292,
        'min_child_weight': 10,
        'random_state': 42
    }
    
    print("Training Champion Model...")
    base_model = xgb.XGBClassifier(**best_params)
    champion_model = BalancedBaggingClassifier(
        estimator=base_model,
        n_estimators=15,
        sampling_strategy='not minority',
        replacement=False,
        random_state=42
    )
    champion_model.fit(X_train, y_train)
    
    # 3. SHAP Calculation (Average across bags)
    print("Calculating SHAP values across the ensemble...")
    all_shap_values = []
    for est in champion_model.estimators_:
        inner_model = est.steps[-1][1]
        explainer = shap.TreeExplainer(inner_model)
        # We use X_test for the explanation
        all_shap_values.append(explainer.shap_values(X_test))
    
    mean_shap_values = np.mean(all_shap_values, axis=0)
    
    # 4. Global Importance
    importance = pd.DataFrame({
        'feature': X.columns,
        'mean_abs_shap': np.abs(mean_shap_values).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)
    
    print("\nTop 10 Champion Features (SHAP Importance):")
    print(importance.head(10))
    
    # 5. Visualization
    plt.figure(figsize=(12, 10))
    shap.summary_plot(mean_shap_values, X_test, show=False)
    plt.title("Champion Model: SHAP Summary (Ensemble Average)")
    plt.tight_layout()
    plt.savefig("karolj/new/champion_shap_summary.png")
    print("\nSHAP summary plot saved to karolj/new/champion_shap_summary.png")

if __name__ == "__main__":
    run_champion_shap()
