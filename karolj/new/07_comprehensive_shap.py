import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.model_selection import GroupShuffleSplit

def get_shap_importance(model, X_test, model_name):
    print(f"Calculating SHAP for {model_name}...")
    
    if isinstance(model, BalancedBaggingClassifier):
        # For bagging, average SHAP values across all estimators
        all_shap_values = []
        for est in model.estimators_:
            # Each est is a Pipeline, the model is the last step
            inner_model = est.steps[-1][1]
            explainer = shap.TreeExplainer(inner_model)
            # Bagging uses undersampled data for training, but we test on full X_test
            # TreeExplainer is generally robust to feature distributions
            all_shap_values.append(explainer.shap_values(X_test))
        
        shap_values = np.mean(all_shap_values, axis=0)
    else:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    
    # Calculate mean absolute SHAP values for importance
    importance = np.abs(shap_values).mean(axis=0)
    df_imp = pd.DataFrame({
        'feature': X_test.columns,
        'shap_importance': importance
    }).sort_values('shap_importance', ascending=False)
    
    # Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False, plot_size=None)
    plt.title(f"SHAP Summary: {model_name}")
    plt.tight_layout()
    plt.savefig(f"karolj/new/shap_{model_name.lower().replace(' ', '_')}.png")
    plt.close()
    
    return df_imp

def run_comprehensive_shap():
    print("Loading data...")
    df = pd.read_csv("for_participants/data/players_quarters_final.csv")
    
    # Feature Engineering
    abs_min_map = {'half_1': 0, 'half_2': 45, 'extra_time_1': 90}
    df['checkpoint_abs_min'] = df['checkpoint_period'].map(abs_min_map) + df['checkpoint_min']
    df['mins_on_pitch'] = df['checkpoint_abs_min'] - df['minute_in']
    
    target = 'scored_after'
    groups = df['fixture_id']
    y = df[target]
    
    # Split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, y, groups=groups))
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Features
    drop_clean = ['player_appearance_id', 'player_id', 'fixture_id', 'date', 'checkpoint', 'checkpoint_period', 'checkpoint_min', 
                  'minute_out', 'subbed', 'jersey_number', 'checkpoint_abs_min', 'minute_in']
    X_clean = pd.get_dummies(df.drop(columns=drop_clean + [target]), columns=['position', 'formation'], drop_first=True)
    X_leaky = pd.get_dummies(df.drop(columns=['player_appearance_id', 'player_id', 'fixture_id', 'date', 'checkpoint', 'checkpoint_period', 'checkpoint_min', target]), 
                            columns=['position', 'formation'], drop_first=True)

    X_c_train, X_c_test = X_clean.iloc[train_idx], X_clean.iloc[test_idx]
    X_l_train, X_l_test = X_leaky.iloc[train_idx], X_leaky.iloc[test_idx]

    results = []

    # 1. Naive Baseline
    m1 = xgb.XGBClassifier(scale_pos_weight=(len(y_train)-sum(y_train))/sum(y_train), random_state=42)
    m1.fit(X_l_train, y_train)
    results.append(get_shap_importance(m1, X_l_test, "Naive Baseline"))

    # 2. Clean Model
    m2 = xgb.XGBClassifier(scale_pos_weight=(len(y_train)-sum(y_train))/sum(y_train), random_state=42)
    m2.fit(X_c_train, y_train)
    results.append(get_shap_importance(m2, X_c_test, "Clean Model"))

    # 3. Balanced Bagging
    m3 = BalancedBaggingClassifier(estimator=xgb.XGBClassifier(random_state=42), n_estimators=10, random_state=42)
    m3.fit(X_c_train, y_train)
    results.append(get_shap_importance(m3, X_c_test, "Balanced Bagging"))

    # 4. Tuned Bagging
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
    m4 = BalancedBaggingClassifier(estimator=xgb.XGBClassifier(**tuned_params), n_estimators=15, random_state=42)
    m4.fit(X_c_train, y_train)
    results.append(get_shap_importance(m4, X_c_test, "Tuned Bagging"))

    # Combine all importances
    model_names = ["Naive", "Clean", "Bagging", "TunedBagging"]
    for i, name in enumerate(model_names):
        results[i].columns = ['feature', f'shap_{name}']
    
    # Merge on feature
    from functools import reduce
    final_imp = reduce(lambda left, right: pd.merge(left, right, on='feature', how='outer'), results)
    final_imp.to_csv("karolj/new/comprehensive_shap_importance.csv", index=False)
    
    print("\nComprehensive SHAP analysis complete. Plots saved to karolj/new/")
    print("\nTop 5 Features comparison:")
    print(final_imp.sort_values('shap_TunedBagging', ascending=False).head(5))

if __name__ == "__main__":
    run_comprehensive_shap()
