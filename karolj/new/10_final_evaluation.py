import pandas as pd
import numpy as np
import xgboost as xgb
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (roc_auc_score, average_precision_score, 
                             precision_score, recall_score, f1_score, confusion_matrix)

def evaluate_model(X_train, y_train, X_test, y_test, model, name):
    model.fit(X_train, y_train)
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    return {
        "Model": name,
        "ROC-AUC": roc_auc_score(y_test, y_probs),
        "PR-AUC": average_precision_score(y_test, y_probs),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn
    }

def run_comprehensive_worth_test():
    print("Loading tactical dataset...")
    df = pd.read_csv("karolj/new/players_conservative_tactical.csv")
    
    # Feature Engineering
    abs_min_map = {'half_1': 0, 'half_2': 45, 'extra_time_1': 90}
    df['checkpoint_abs_min'] = df['checkpoint_period'].map(abs_min_map) + df['checkpoint_min']
    df['mins_on_pitch'] = df['checkpoint_abs_min'] - df['minute_in']
    team_group = df.groupby(['fixture_id', 'is_home', 'checkpoint'])
    df['team_last15_shots'] = team_group['last15_shots'].transform('sum')
    
    target = 'scored_after'
    groups = df['fixture_id']
    y = df[target]
    
    # Feature Sets
    base_physicals = [
        'last15_sprints', 'last15_hsr', 'last15_distance', 'last15_mean_max_speed', 'last15_peak_speed',
        'last15_shots', 'last15_shots_on_target', 'last15_shots_under_press', 'last15_shots_top_third',
        'cumul_sprints', 'cumul_hsr', 'cumul_distance', 'cumul_mean_max_speed', 'cumul_peak_speed',
        'cumul_shots', 'cumul_shots_on_target', 'cumul_shots_under_press', 'cumul_shots_top_third'
    ]
    momentum_features = ['mins_on_pitch', 'team_last15_shots']
    tactical_features = ['l15_top_third_pass_acc', 'l15_press_forward_rate', 'cumul_explosivity_ratio']
    categorical_features = [c for c in df.columns if 'position_' in c or 'formation_' in c] # Will be added via dummies
    
    # Prep dummies
    X_all_dummies = pd.get_dummies(df.drop(columns=['player_appearance_id', 'player_id', 'fixture_id', 'date', 'checkpoint', 'checkpoint_period', 'checkpoint_min', 'minute_out', 'subbed', 'jersey_number', 'checkpoint_abs_min', 'minute_in', 'period_val', target]), 
                                  columns=['position', 'formation'], drop_first=True)
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, y, groups=groups))
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Define Models to test
    tuned_params = {
        'learning_rate': 0.10129, 'max_depth': 8, 'n_estimators': 149, 'random_state': 42, 'eval_metric': 'aucpr'
    }
    
    results = []
    
    # 1. Base Model (Original Features Only)
    X1 = X_all_dummies.drop(columns=momentum_features + tactical_features)
    results.append(evaluate_model(X1.iloc[train_idx], y_train, X1.iloc[test_idx], y_test, 
                                 xgb.XGBClassifier(scale_pos_weight=(len(y_train)-sum(y_train))/sum(y_train), **tuned_params), 
                                 "M1: Base Physicals"))
    
    # 2. Base + Momentum
    X2 = X_all_dummies.drop(columns=tactical_features)
    results.append(evaluate_model(X2.iloc[train_idx], y_train, X2.iloc[test_idx], y_test, 
                                 xgb.XGBClassifier(scale_pos_weight=(len(y_train)-sum(y_train))/sum(y_train), **tuned_params), 
                                 "M2: M1 + Momentum"))
    
    # 3. Base + Momentum + Tactical
    X3 = X_all_dummies
    results.append(evaluate_model(X3.iloc[train_idx], y_train, X3.iloc[test_idx], y_test, 
                                 xgb.XGBClassifier(scale_pos_weight=(len(y_train)-sum(y_train))/sum(y_train), **tuned_params), 
                                 "M3: M2 + Tactical"))
    
    # 4. The Ensemble (Tuned Bagging with all features)
    results.append(evaluate_model(X3.iloc[train_idx], y_train, X3.iloc[test_idx], y_test, 
                                 BalancedBaggingClassifier(estimator=xgb.XGBClassifier(**tuned_params), n_estimators=15, random_state=42), 
                                 "M4: Full Ensemble (Bagging)"))
    
    # 5. Ablation: Just the 3 Tactical Features (Worth test)
    # How much info do these 3 alone carry?
    X5 = X_all_dummies[tactical_features]
    results.append(evaluate_model(X5.iloc[train_idx], y_train, X5.iloc[test_idx], y_test, 
                                 xgb.XGBClassifier(scale_pos_weight=(len(y_train)-sum(y_train))/sum(y_train), **tuned_params), 
                                 "Ablation: Only 3 Tactical Features"))

    # Convert to DF and print
    res_df = pd.DataFrame(results)
    print("\n--- Comprehensive Performance & Worth Summary ---")
    print(res_df.to_string(index=False))
    
    res_df.to_csv("karolj/new/final_worth_comparison.csv", index=False)
    print("\nDetailed results saved to karolj/new/final_worth_comparison.csv")

if __name__ == "__main__":
    run_comprehensive_worth_test()
