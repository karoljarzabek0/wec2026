import pandas as pd
import numpy as np
import xgboost as xgb
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, confusion_matrix

def run_final_comparison():
    print("Loading data...")
    df = pd.read_csv("for_participants/data/players_quarters_final.csv")
    
    # 1. Shared Feature Engineering
    abs_min_map = {'half_1': 0, 'half_2': 45, 'extra_time_1': 90}
    df['checkpoint_abs_min'] = df['checkpoint_period'].map(abs_min_map) + df['checkpoint_min']
    df['mins_on_pitch'] = df['checkpoint_abs_min'] - df['minute_in']
    
    team_group = df.groupby(['fixture_id', 'is_home', 'checkpoint'])
    df['team_last15_shots'] = team_group['last15_shots'].transform('sum')
    df['team_last15_sprints'] = team_group['last15_sprints'].transform('sum')
    df['team_last15_distance'] = team_group['last15_distance'].transform('sum')
    
    # Target and Groups
    target = 'scored_after'
    groups = df['fixture_id']
    y = df[target]
    
    # Common Drops
    common_drops = ['player_appearance_id', 'player_id', 'fixture_id', 'date', 'checkpoint', 'checkpoint_period', 'checkpoint_min', 'minute_out', 'subbed', 'jersey_number', 'checkpoint_abs_min', 'minute_in']

    # --- MODEL A: Previous Champion (Full Features) ---
    X_a = pd.get_dummies(df.drop(columns=common_drops + [target]), columns=['position', 'formation'], drop_first=True)

    # --- MODEL B: Simplified Champion (Requested) ---
    df_b = df.copy()
    df_b['is_attacker_mid'] = df_b['position'].isin(['A', 'M']).astype(int)
    X_b = df_b.drop(columns=common_drops + ['position', 'formation', target])
    
    # Shared Split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, y, groups=groups))
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Shared Parameters
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

    def train_eval(X, name):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        base = xgb.XGBClassifier(**best_params)
        model = BalancedBaggingClassifier(estimator=base, n_estimators=15, random_state=42)
        model.fit(X_train, y_train)
        
        y_probs = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        return {
            "Model": name,
            "ROC-AUC": roc_auc_score(y_test, y_probs),
            "PR-AUC": average_precision_score(y_test, y_probs),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn
        }

    res_a = train_eval(X_a, "Previous Champion (Full)")
    res_b = train_eval(X_b, "Simplified Champion (Cleaned)")

    results = pd.DataFrame([res_a, res_b])
    print("\n--- Side-by-Side Comparison ---")
    print(results.to_string(index=False))

if __name__ == "__main__":
    run_final_comparison()
