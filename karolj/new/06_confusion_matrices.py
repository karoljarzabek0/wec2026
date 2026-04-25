import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_cm(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f"Confusion Matrix: {title}")
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f"karolj/new/{filename}")
    plt.close()

def generate_all_cms():
    print("Loading data...")
    df = pd.read_csv("for_participants/data/players_quarters_final.csv")
    
    # 1. Feature Prep for Leaky Baseline
    abs_min_map = {'half_1': 0, 'half_2': 45, 'extra_time_1': 90}
    df['checkpoint_abs_min'] = df['checkpoint_period'].map(abs_min_map) + df['checkpoint_min']
    df['mins_on_pitch'] = df['checkpoint_abs_min'] - df['minute_in']
    
    target = 'scored_after'
    groups = df['fixture_id']
    y = df[target]
    
    # Leaky Features
    X_leaky = pd.get_dummies(df.drop(columns=['player_appearance_id', 'player_id', 'fixture_id', 'date', 'checkpoint', 'checkpoint_period', 'checkpoint_min', target]), 
                            columns=['position', 'formation'], drop_first=True)
    
    # Clean Features (dropping leakage)
    drop_clean = ['player_appearance_id', 'player_id', 'fixture_id', 'date', 'checkpoint', 'checkpoint_period', 'checkpoint_min', 
                  'minute_out', 'subbed', 'jersey_number', 'checkpoint_abs_min', 'minute_in']
    X_clean = pd.get_dummies(df.drop(columns=drop_clean + [target]), columns=['position', 'formation'], drop_first=True)
    
    # Split (Same for all)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, y, groups=groups))
    
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # --- MODEL 1: Naive Baseline (Leaky) ---
    print("Running Naive Baseline...")
    X_l_train, X_l_test = X_leaky.iloc[train_idx], X_leaky.iloc[test_idx]
    model1 = xgb.XGBClassifier(scale_pos_weight=(len(y_train)-sum(y_train))/sum(y_train), random_state=42)
    model1.fit(X_l_train, y_train)
    plot_cm(y_test, model1.predict(X_l_test), "Naive Baseline (Leaky)", "cm_01_baseline.png")

    # --- MODEL 2: Clean Model ---
    print("Running Clean Model...")
    X_c_train, X_c_test = X_clean.iloc[train_idx], X_clean.iloc[test_idx]
    model2 = xgb.XGBClassifier(scale_pos_weight=(len(y_train)-sum(y_train))/sum(y_train), random_state=42)
    model2.fit(X_c_train, y_train)
    plot_cm(y_test, model2.predict(X_c_test), "Clean Model", "cm_02_clean.png")

    # --- MODEL 3: Balanced Bagging ---
    print("Running Balanced Bagging...")
    base_xgb = xgb.XGBClassifier(random_state=42)
    model3 = BalancedBaggingClassifier(estimator=base_xgb, n_estimators=10, random_state=42)
    model3.fit(X_c_train, y_train)
    plot_cm(y_test, model3.predict(X_c_test), "Balanced Bagging", "cm_03_bagging.png")

    # --- MODEL 4: Tuned Bagging ---
    print("Running Tuned Bagging...")
    tuned_params = {
        'learning_rate': 0.10129919477568608,
        'max_depth': 8,
        'n_estimators': 149,
        'subsample': 0.7807733062526079,
        'colsample_bytree': 0.7754409018439148,
        'gamma': 0.003619980766962292,
        'min_child_weight': 10,
        'random_state': 42
    }
    base_tuned = xgb.XGBClassifier(**tuned_params)
    model4 = BalancedBaggingClassifier(estimator=base_tuned, n_estimators=15, random_state=42)
    model4.fit(X_c_train, y_train)
    plot_cm(y_test, model4.predict(X_c_test), "Tuned Bagging", "cm_04_tuned_bagging.png")

    print("\nConfusion matrices saved to karolj/new/")

if __name__ == "__main__":
    generate_all_cms()
