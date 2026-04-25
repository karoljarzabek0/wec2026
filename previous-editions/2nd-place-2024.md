# Review of WEC 2024 2nd Place Submission: "Politics, Institutions and (Random) Forests?"

**Team:** Maximum (Victory) Likelihood EsTeamation  
**Focus:** Drivers of COVID-19 vaccination rates in Poland.

---

## 1. Initial Ideas & Hypotheses
The team's research centered on identifying why Poland's vaccination rates lagged behind other CEE countries despite similar economic conditions. Their core hypotheses explored:
*   **Historical Legacy:** Do the 19th-century partitions of Poland (Prussian, Russian, Austrian) still influence health behaviors today?
*   **Political Drivers:** How do political preferences (left-wing vs. right-wing support) correlate with vaccine hesitancy or enthusiasm?
*   **Socio-Demographics:** The impact of urbanization, age structure (elderly vs. youth), and gender.
*   **Public Services:** Does the provision of public goods (healthcare, education, infrastructure) ease vaccine access?
*   **Social Learning:** Does exposure to abnormal COVID-19 mortality drive people to vaccinate?

## 2. Dataset Exploration & Enhancement
The team started with the provided WEC dataset and significantly enhanced it:
*   **Local Data Bank (BDL GUS):** Added agricultural variables (rural taxes, farm sizes), unemployment by age groups, and public service indicators.
*   **National Electoral Commission (PKW):** Merged 2019 parliamentary election results to map political preferences.
*   **Spatial Data:** Calculated centroids (latitude/longitude) for all municipalities to enable spatial modeling.
*   **Feature Engineering:** 
    *   **Excess Mortality Index:** Calculated as `Mean(2010-2019 deaths) - 2020 deaths` to capture pandemic exposure.
    *   **Public Goods Provision Index:** Created using Principal Component Analysis (PCA) to reduce 6 dimensions of public services into a single score.
    *   **Urbanization Categories:** Created a categorical variable `type` based on population thresholds.

## 3. Managing Abnormal Data
### Missing Data
The team used a **Hierarchical Mean Imputation** strategy in R:
1.  First, they imputed missing values using the **mean of the county (powiat)** the municipality belongs to.
2.  If data was still missing (i.e., the entire county was missing data), they used the **overall mean** of the variable across all municipalities.
3.  This was applied to variables like `healthcare_advices`, `forests_area`, `children_3_5_in_kindergartens`, etc.

### Outliers
There is no explicit evidence in the code or paper of standard outlier removal (like Winsorization). The team relied on the robustness of models like Random Forest and XGBoost to handle non-linearities and potential outliers naturally.

## 4. Models Used

### Econometric Models (Causal Inference)
*   **OLS (Ordinary Least Squares):** Used as a baseline to detect collinearity (VIF) and general trends.
*   **Spatial Regression (SEM & SAR):** 
    *   **SEM (Spatial Error Model):** Addressed spatial dependence in the error terms.
    *   **SAR (Spatial Autoregressive Model):** Accounted for the "spillover" effect where vaccination in one municipality affects neighbors.
*   **Instrumental Variables (IV / 2SLS / GMM):** 
    *   Addressed the endogeneity of political preferences.
    *   **Instrument:** Support for left-wing parties in the rest of the county (shift-share design).
*   **Regression Discontinuity Design (RDD):** Used to test the "partition" effect specifically at the borders of historical partitions.

### Machine Learning Models (Prediction & Importance)
*   **K-Nearest Neighbors (KNN)**
*   **Support Vector Machines (SVM)**
*   **Random Forest (RF)**
*   **XGBoost:** The best performing model (lowest RMSE).
*   **Explainable ML (SHAP):** Used to "open the black box" of XGBoost and interpret feature importance.

### Deep Learning Models
*   **None Used.** The team prioritized interpretability and econometric rigor over deep learning, which is often less suited for tabular municipality-level data with ~2,500 observations.

## 5. Results & Model Behavior

### Key Findings:
*   **Politics:** Surprisingly, it wasn't just "right-wing hesitancy" but rather **"left-wing enthusiasm"** (SLD/KO voters) that strongly drove vaccination rates.
*   **Geography:** Higher vaccination in the North, lower in the South-East.
*   **Demographics:** A high share of elderly (60-84) increased vaccination rates, while a high share of young adults (20-29) decreased them.
*   **The "Partition" Myth:** While OLS/SEM suggested some partition effects, the **RDD results were insignificant**. The team concluded that historical partitions are less relevant than current political/social factors.
*   **Exposure:** Excess mortality from 2020 had *no* significant effect on vaccination uptake (social learning was weak).

### Unexpected Behaviors & Management:
*   **Spatial Dependence:** The Moran's I test was highly significant (0.71), forcing a pivot from simple OLS to complex Spatial Models.
*   **Model Performance:** XGBoost significantly outperformed linear models (RMSE 3.47 vs others), indicating strong non-linear relationships that OLS could not capture.
*   **RDD Contradiction:** The team's initial hypothesis about Prussian vs. Russian partitions was not supported by RDD. They managed this by honestly reporting the lack of significance and shifting focus to political drivers.

## 6. Final Presentation
The team presented their work in a professional academic format:
*   **Paper:** A 30+ page PDF with JEL classifications, formal equations, and rigorous citations.
*   **Visuals:** Extensive use of maps (Clustering, SHAP maps) and SHAP dependence plots to make complex ML results intuitive.
*   **Code:** Well-organized R and Python scripts/notebooks.

---

## 7. Personal Commentary (Disclaimer: AI Opinion)
*Disclaimer: The following are observations and opinions of the Gemini CLI assistant and do not represent the views of the original authors.*

1.  **Strengths:** The team's integration of **Spatial Econometrics** with **Machine Learning (SHAP)** is exceptionally strong. Most teams pick one or the other; doing both allows for both causal explanation and high-performance prediction.
2.  **Weaknesses:** The handling of **missing data** (simple mean imputation) could be improved. In a dataset with high spatial correlation, **Kriging** or **Spatial Imputation** would have been more accurate than a county-level mean.
3.  **Innovation:** Using the **rest-of-county political support** as an instrument for local support is a clever adaptation of the shift-share design commonly used in labor economics.
4.  **Aesthetics:** Their maps (especially the K-means clustering of vaccination rates) are very effective at showing the "Eastern Poland" vs "Western Poland" divide that persists despite the RDD findings.

---
