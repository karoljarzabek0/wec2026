  ---

  Warsaw Econometric Challenge 2025: Winning Team Submission Report
  Team Name: p-value abusers
  Task: Management and prediction of traffic data using diverse spatial and temporal datasets.

  ---

##  1. Project Overview & Initial Ideas
  The team's approach was characterized by a heavy emphasis on spatial feature engineering and a dual-modeling strategy combining econometric interpretability with machine learning
  predictive power.

  Initial Strategies:
   * Spatial Contextualization: Instead of treating stations as isolated points, the team reconstructed the "neighborhood" of each station.
   * Target Transformation: Early EDA suggested that traffic data was highly skewed, leading the team to experiment with square root transformations (sqrt(traffic)) to stabilize variance
     in linear models.
   * H3 Hexagonal Hierarchical Indexing: Use of Uber’s H3 library for efficient spatial indexing and joining of disparate datasets (signals, POIs, buildings).

  ---

## 2. Dataset Exploration & Management
  The team managed a complex "star-schema" of data centered around traffic stations.

  Spatial Aggregation (The "Secret Sauce"):
  The team used DuckDB and Polars to perform high-performance spatial joins. They aggregated features within various radii (0.5km, 1km, 1.5km, 2km, and 3km) around each traffic station:
   * Signals Data: Aggregated signal counts per day/hour.
   * Buildings: Summed building counts by type (Residential, Service, Work).
   * Demography: Population counts partitioned by age and gender.
   * POIs (Points of Interest): Counts of businesses, attractions, and services (e.g., "automotive", "eat_and_drink").
   * Roads: Road intensity and infrastructure metrics.
   * Meteo: Weather conditions (temperature, precipitation).

  Handling Abnormal Data:
   * Missing Data: For POIs and building counts, the team used a logical "Zero Assumption" (fillna(0)), assuming that if a category wasn't listed near a station, it didn't exist there.
   * Outliers/High Traffic: The team focused on the 95th percentile (quantile(0.95)) of traffic, identifying that "normal" models often fail to capture the peaks which are most critical
     for traffic management.
   * Station-Specific Behavior: They performed granular EDA on specific stations (e.g., Station 2301, 3332), identifying unique local patterns that global models might miss.

  ---

##  3. Modeling Approach
  The team utilized three distinct classes of models:

  A. Econometric Models (R - mgcv/stats)
  They used Generalized Additive Models (GAMs) and Linear Models (lm).
   * Key Features: s(hour) (smooths for time of day), factor(station_id), and interactions like s(hour, by = signal_count).
   * Purpose: These were used to understand the "physics" of the traffic—how specific variables like signal counts or holidays non-linearly impact flow.

  B. Machine Learning Models (Python - XGBoost/RandomForest)
  For the final competition submission, the team leaned on ensemble methods.
   * Models: XGBRegressor and RandomForestRegressor.
   * Hyperparameter Tuning: They performed extensive depth tuning (max_depth from 2 to 11).
   * Validation: 5-fold cross-validation was used to ensure robustness.
   * Results for RF (depth=9):
       * Avg RMSE (Val): ~1319.5
       * Avg R² (Val): ~0.453
       * Note: The discrepancy between Train R² (~0.98) and Val R² (~0.45) indicates high model complexity, likely intentional to capture specific high-traffic peaks.

  C. Deep Learning
  While the repository contains mentions of PyTorch/TensorFlow in search strings, the core winning logic appears to be dominated by the refined spatial features in XGBoost and GAMs.

  ---

##  4. Problem Solving & Model Diagnostics
  The team did not just "fit and predict"; they used a rigorous feedback loop:
   1. Residual Analysis: They took residuals from their GAM models and plotted them against time and space to see where the model was "blind."
   2. Spatial Correlation: Used corrplot in R to check if residuals were spatially correlated, which helped them refine the radii for their spatial joins.
   3. Visual Validation: Extensive use of val_preds_vs_true plots to visually inspect if the models were under-predicting the 95th percentile peaks.

  ---

##  5. Final Presentation & Results
  The team's final output focused on the q95 (95th percentile) traffic prediction, which is crucial for infrastructure stress testing. They presented their results with:
   * Clear diagnostic plots (Residuals, QQ-plots).
   * Station-level performance reports.
   * Comparison of model depths to justify the final selection (Depth 9 was a common sweet spot).

  ---

##  6. Gemini's Technical Commentary (Disclaimer: AI Opinion)
   * Spatial Engineering Excellence: In my opinion, the team won because of their data join strategy. Using H3 to create multi-radius features (0.5km to 3km) allowed the model to
     understand both the immediate environment (e.g., a traffic light right next to the station) and the macro environment (e.g., being in a high-density residential zone).
   * Tooling Synergies: The choice of DuckDB + Polars is a "modern stack" move that likely gave them a massive speed advantage in feature engineering compared to teams using standard
     Pandas.
   * Econometrics as a Compass: I find their use of GAMs particularly clever. Instead of jumping straight to "black box" XGBoost, they used GAMs to verify that their features made sense
     (e.g., signal counts having a smooth relationship with traffic), which likely prevented them from including "noise" features.
   * Overfitting Risk: The gap between Train and Validation R² is significant. However, in traffic forecasting, capturing the variance of peak hours is often more important than the mean
     error, and their depth-tuning reports show they were very aware of this trade-off.

  ---

