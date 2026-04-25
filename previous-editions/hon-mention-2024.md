# Analysis of the WEC 2024 Submission: "P-value Abusers"

This document provides a comprehensive analysis of the submission by the "P-value Abusers" team for the 2024 Warsaw Econometric Challenge. The task involved analyzing the drivers of COVID-19 vaccination rates across Polish municipalities.

---

## 1. Initial Ideas and Hypotheses
The team grounded their research in the **5C model of vaccine hesitancy**, which identifies five key psychological antecedents:
*   **Confidence:** Trust in vaccine effectiveness, safety, and the health system.
    *   *Hypothesis:* Higher voter turnout (proxy for trust in authorities) and specific political affiliations correlate with vaccination rates.
*   **Complacency:** Perception of the risk posed by the disease.
    *   *Hypothesis:* Municipalities with more seniors (higher risk) have higher rates; those with more young people (lower risk) have lower rates.
*   **Convenience:** Physical availability and ease of access.
    *   *Hypothesis:* Proximity to vaccination sites and car ownership per 1000 inhabitants positively correlate with rates.
*   **Calculation:** Individual engagement in extensive information searching.
    *   *Hypothesis:* Higher education levels correlate with higher vaccination rates.
*   **Collective Responsibility:** Willingness to protect others.

**Other Factors:**
*   Population density, income per capita (PIT revenue), and historical partitions of Poland (Prussian, Russian, Austrian) were hypothesized to play significant roles.
*   **Spatial Dependencies:** The team hypothesized that a municipality's vaccination rate is influenced by its neighbors.

---

## 2. Dataset Exploration and Management of Abnormal Data

### Data Sources
*   **Organizer Data:** Vaccination rates (2021) and county-level socio-economic data (2020).
*   **External Data:** 2019 Parliamentary election results (PKW), shapefiles for spatial analysis, and a custom dataset of vaccination points (March 2021).

### Managing Missing Data and Outliers
*   **KNN Imputation:** The team used `KNNImputer` (n=5) to handle missing values in features like `healthcare_advices_ratio_total`, `forests_area`, and `children_3_5_in_kindergartens`.
*   **The "Warsaw Fix":** They identified a specific issue with Warsaw (TERYT 1465011) where political data was missing. They manually filled this using aggregated data for the city and set fixed values for distance to vaccination points.
*   **Data Consistency:** They cleaned numerical strings (removing spaces) and handled "county-level" features by mapping them to municipalities.

### Feature Engineering
*   **Dependent Variable:** The `percent_vaccinated` was divided by 100 and transformed using the **logit function** to make it unbounded, meeting linear regression assumptions.
*   **Transformations:** 
    *   Logarithmic: `revenues_per_capita_PIT`, `population_density`.
    *   Square Root: `SLD_percent`, `min_dist`.
    *   Polynomial: `cars_per_1000_persons` (2nd degree), `percent_over_60` (2nd degree).
    *   Interactions: `PO_percent : frekwencja_wyborcza` (Political support x Turnout).

---

## 3. Econometric Models

### Linear Regression (The MVP)
*   **Approach:** Used forward and backward feature selection based on F-tests and AIC.
*   **Findings:** Political views (PO, SLD positive; Konfederacja negative) and age (>60) were strong predictors. 
*   **Problem:** Diagnostic plots (QQ-plot) showed heavy tails, and more importantly, **Moran's I test** indicated significant spatial autocorrelation in residuals (p < 0.001), violating the independence assumption.

### Spatial Models (SARAR/SAC)
*   **Approach:** To solve the spatial dependence problem, they implemented a **Spatial Autoregressive with Autoregressive Conditional Heteroskedasticity (SARAR)** model.
*   **Implementation:** They used a weight matrix based on reversed distances between municipalities (Haversine formula).
*   **Results:** This model successfully accounted for spatial clusters. The residuals became random (Moran's I p-value ~ 0.999), proving the model's validity.

---

## 4. Machine Learning Models

### XGBoost
*   **Motivation:** To capture complex non-linear dependencies without manual feature transformation.
*   **Setup:** Used a standard XGBRegressor pipeline with One-Hot Encoding for categorical variables (`partitions`, `type_of_municipality`).
*   **Explainability (SHAP):** The team prioritized interpretability over raw performance. They used **SHAP (SHapley Additive exPlanations)** to visualize feature importance.

---

## 5. Deep Learning Models
*   **Findings:** Based on the repository analysis, **no deep learning models (Neural Networks, CNNs, etc.) were used**. The team stayed within the realms of classical econometrics and boosted trees (XGBoost), likely due to the tabular nature of the data and the emphasis on statistical inference.

---

## 6. Results and Model Behavior

### Key Findings
*   **Political Factors:** The strongest drivers were political. Support for "progressive" parties (PO, SLD) and high voter turnout positively influenced vaccination. Support for Konfederacja had a negative impact.
*   **Demographics:** The percentage of seniors (>60) had a non-linear effect (positive initially, then leveling off).
*   **Spatial Context:** A "neighbor effect" was confirmed; municipalities surrounded by high-vaccination areas tended to have higher rates themselves.
*   **Surprises:** Education level was not always a statistically significant direct predictor in the spatial model, as its effect was often captured by other socio-economic variables.

### Unexpected Behaviors & Solutions
*   **Multicollinearity:** They found high correlation between `PIS_percent` and `PO_percent` (-0.88), leading to the exclusion of `PIS_percent` to stabilize the model.
*   **Spatial Autocorrelation:** Initially, the linear model failed diagnostics. The transition to a SARAR model was their primary technical solution.

---

## 7. Presentation
The team presented their work via:
1.  **A detailed PDF paper:** "The drivers of the level of COVID-19 vaccination in Poland," covering background, literature, and methodology.
2.  **Visual Diagnostics:** Partial residual plots, VIF tables, and SHAP beeswarm plots.
3.  **Spatial Maps:** LISA (Local Indicators of Spatial Association) maps showing vaccination clusters and outlier residuals.

---

## 8. Expert Commentary (Gemini CLI Opinion)
*Disclaimer: These comments represent my own technical assessment and not the views of the original team.*

*   **Strengths:** 
    *   The use of the **5C model** provides a very strong theoretical foundation, which is often missing in "pure" data science submissions. 
    *   The transition from a failing Linear Model to a **SARAR model** shows a high level of econometric maturity and a proper understanding of spatial data. 
    *   The manual fix for Warsaw shows attention to detail in data cleaning.
*   **Weaknesses/Risks:**
    *   The use of **KNNImputer** on spatial data can be risky if not carefully tuned, as it might "smooth" out local variations too much.
    *   The exclusion of PiS support due to correlation with PO support is a standard fix, but they could have explored a "Political Polarization" index instead.
    *   The absence of a temporal component (as they admitted in the discussion) limits the model's ability to explain the *dynamic* of the vaccination campaign.
*   **Overall:** It was a technically sound, theoretically grounded submission that correctly identified and solved the "spatial problem" common in regional data.
