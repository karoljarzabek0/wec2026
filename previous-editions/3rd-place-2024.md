# Warsaw Econometric Challenge 2024 - 3rd Place Report: Gradient Descendants

## Disclaimer
*This report is an analysis of the work done by the "Gradient Descendants" team for the Warsaw Econometric Challenge 2024. The comments and opinions expressed herein are those of the analyzing agent and do not necessarily reflect the views or intentions of the original authors.*

---

## 1. Project Overview
The "Gradient Descendants" team secured 3rd place in the 2024 Warsaw Econometric Challenge. Their task involved predicting COVID-19 vaccination rates at the municipality level in Poland. The repository demonstrates a comprehensive approach combining spatial econometrics with state-of-the-art machine learning techniques.

## 2. Dataset Exploration & Initial Ideas
### Initial Approach
The team integrated data from multiple levels:
- **Municipality-level data**: Population density, urbanization rate, healthcare access, and political election results.
- **County-level data**: Average wages, education levels, and healthcare resources (doctors/beds).
- **Spatial Data**: Integration of shapefiles to account for geographical dependencies between municipalities.

### Feature Engineering
- **Target Transformation**: Instead of predicting raw percentages, the team used a **logit transformation**: `log(p / (1 - p))`, where `p` is the proportion of vaccinated individuals. This is a standard approach for bounded variables to map them to an infinite range, which is more suitable for regression models.
- **Demographics**: Aggregated population into age-specific buckets (e.g., % below 25, % 25-60, % over 60).
- **Political Alignment**: Included results for major political parties (PiS, KO, PSL, SLD, Konfederacja) and election turnout. This proved to be a critical factor in explaining vaccination variance.

## 3. Managing Abnormal Data
### Missing Data
- Missing values in specific columns (e.g., `tourists_per_1000_persons`) were identified and handled during the preprocessing phase.
- Some features with high missingness or low relevance were dropped to reduce noise.

### Outliers and Feature Selection
- The team performed a rigorous feature selection, dropping over 30 columns including detailed infrastructure stats (e.g., bicycle paths, specific unemployment sub-types) that did not contribute significantly to the predictive power.
- **Spatial Lag**: They calculated the "Spatial Lag" of vaccination rates (average of neighboring municipalities) using Queen contiguity. This allowed the models to capture "neighborhood effects."

## 4. Models Used

### Econometric Models
The team explored spatial econometric models:
- **SAR (Spatial Autoregressive Model)**: Assumes the dependent variable in one location is affected by the dependent variable in neighboring locations.
- **SDM (Spatial Durbin Model)**: Extends SAR by also including spatial lags of independent variables.

### Machine Learning Models
Three tree-based regression models were implemented:
1. **Random Forest (RF)**
2. **XGBoost**
3. **Gradient Boosting Machine (GBM)**

## 5. Results & Model Performance

### Validation Strategy
The team used a **Leave-One-Voivodeship-Out** cross-validation strategy. This is highly recommended for spatial data as it tests the model's ability to generalize to entirely new regions, preventing overfitting to local spatial clusters.

### Comparative Performance (RMSE in Logit Space)
| Model | Avg. Eval RMSE | Out-of-Sample Performance |
| :--- | :--- | :--- |
| **GBM** | ~0.135 | **Best overall performance** |
| **Random Forest** | ~0.141 | Good stability |
| **XGBoost** | ~0.146 | Slightly higher error |
| **SAR / SDM** | ~0.133 (In-sample) | **Failed out-of-sample (RMSE 2.0 - 4.0)** |

**Key Finding**: While the econometric models (SAR/SDM) showed excellent in-sample fit, their out-of-sample performance was poor. The tree-based ML models, specifically **GBM**, proved far superior for prediction on unseen geographical regions.

## 6. Unexpected Problems & Solutions
- **Econometric Failure**: The team observed that spatial econometric models struggled with out-of-sample prediction. This is a common issue when the spatial weight matrix (`W`) for the test set is disconnected from the training set, or when regional differences (non-stationarity) are not fully captured by the linear form.
- **Political Factor**: The SHAP analysis revealed that political affiliation was among the strongest predictors of vaccination rates. This highlighted that COVID-19 vaccination was highly correlated with socio-political views in Poland.

## 7. Model Interpretation (XAI)
The team used **SHAP (SHapley Additive exPlanations)** to interpret their best model (GBM).
- **Top Positive Drivers**: Support for Koalicja Obywatelska (KO), Urbanization Rate, PIT revenues, and higher % of population over 60.
- **Top Negative Drivers**: Support for Prawo i Sprawiedliwość (PiS) and Konfederacja, higher % of population below 25.
- **Spatial Lag**: The vaccination rate of neighbors was a very strong positive predictor, confirming the importance of spatial effects.

## 8. Agent's Comments (Opinion)
*Opinion Disclaimer: These are independent observations.*

1. **Robust Validation**: The use of leave-one-voivodeship-out cross-validation is the "gold standard" for this type of challenge. It accurately simulates the real-world task of predicting data for a region where you have no labels.
2. **Feature Fusion**: Merging county-level socio-economic data with municipality-level political data was a clever way to bypass the lack of certain indicators at the more granular level.
3. **Logit Transformation**: Many teams fail to transform the target variable. By using the logit transformation, this team ensured their predictions stayed within a mathematically sound range for vaccination proportions.
4. **Spatial Lag in ML**: Manually calculating spatial lags and feeding them into tree-based models (which don't natively understand geography) is a highly effective "spatial ML" hybrid approach.
5. **Political Insight**: Recognizing and quantifying the link between election results and public health behavior was a key differentiator for this team.

---
