import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

def run_baseline_model():
    print("Loading core dataset...")
    df = pd.read_csv("for_participants/data/players_quarters_final.csv")
    
    # Identify target and potential features
    target = 'scored_after'
    
    # Drop identifiers and metadata
    metadata_cols = ['player_appearance_id', 'player_id', 'fixture_id', 'date', 'checkpoint', 'checkpoint_period', 'checkpoint_min']
    
    # Prepare X and y
    y = df[target]
    X_raw = df.drop(columns=metadata_cols + [target])
    
    # Convert categorical variables
    X = pd.get_dummies(X_raw, columns=['position', 'formation'], drop_first=True)
    
    # Convert boolean to int
    if 'subbed' in X.columns:
        X['subbed'] = X['subbed'].astype(int)
    
    # Grouped Train-Test Split (by fixture_id)
    print("Performing Grouped Train-Test Split...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=df['fixture_id']))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    print(f"Train set size: {len(X_train)}")
    print(f"Test set size:  {len(X_test)}")
    
    # Initialize and train XGBoost
    print("Training XGBoost...")
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='aucpr'
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # Evaluation
    roc_auc = roc_auc_score(y_test, y_probs)
    pr_auc = average_precision_score(y_test, y_probs)
    
    print("\n--- Model Performance ---")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # SHAP Analysis
    print("\nRunning SHAP Analysis...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Output top 10 features by SHAP
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Features by SHAP Importance:")
    print(feature_importance.head(10))
    
    # Save results
    feature_importance.to_csv("karolj/new/shap_importance.csv", index=False)
    
    # Save a summary plot (optional, but good for local review)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig("karolj/new/shap_summary.png")
    print("\nSHAP summary plot saved to karolj/new/shap_summary.png")

if __name__ == "__main__":
    run_baseline_model()
