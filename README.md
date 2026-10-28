# Stress Detection using Machine Learning

One-page summary: open `poster.html` in a browser.

## About the Dataset

This project uses a **synthetic dataset** created from training GANs on the original WESAD dataset. The synthetic data has:

- Data for **10,000 subjects** instead of just 15 in the original dataset.
- Labels are simplified to **only two classes: stress and non-stress**.

Original WESAD dataset: [link](https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download)

Dataset used for training and testing in this project:  
[Google Drive link](https://drive.google.com/file/d/1GA3uIod8d_jHtk2IIwCEYX5AmmNkTaTD/view?usp=sharing)

We did some preprocessing on this dataset, such as handling missing values, skewness, and outliers (in different ways as discussed in stage 2).

---

## Project Workflow

### 1. **Exploratory Data Analysis (EDA)**

- **Checked dataset shape, columns, data types**, and null values.  
- Visualised **label distribution** to confirm balance between stress and non-stress classes.
- **Correlation analysis** was done to check which features are strongly correlated.  
- **Histograms and boxplots** were plotted to understand data distributions and outliers for features like:
  - ACC_x, ACC_y, ACC_z
  - ECG
  - EMG
  - GSR
  - HR
  - RESP
  - TEMP

- **Skewness was reduced** using suitable transformations (log or square root as per feature distribution).

- Outliers were handled by **clipping based on quantile ranges** to avoid extreme distortions.

---

### 2. **Data Preprocessing**

- **Dropped irrelevant columns** if any (noted in code but minimal).  
- **StandardScaler** was used to scale features before feeding into models.  
- The dataset was split into **train-test sets (80-20 split)** with stratification to maintain label balance.

---

### 3. **Modelling**

Multiple models were tested:

#### Logistic Regression
- Baseline simple model.  
- Achieved an accuracy of **~93%**.

#### Random Forest Classifier
- Performed better than logistic regression.  
- Accuracy improved to **~98.2%** with strong F1-score.

#### XGBoost Classifier
- Slightly better than Random Forest on test set with accuracy of **~98.4%**.  
- Took longer to train but generalised well.

#### LightGBM Classifier
- Achieved similar performance to XGBoost with accuracy of **~98.3%**.
- Fast training and prediction times.

---

### 4. **Evaluation Metrics**

For all models, we evaluated:

- **Accuracy**  
- **Precision**  
- **Recall**  
- **F1-score**  
- **Confusion matrix**

**ROC-AUC curves** were also plotted for each classifier to visually compare performance.

---

## Results Summary

| Model               | Accuracy (%) | Remarks                       |
|---------------------|--------------|-------------------------------|
| Logistic Regression | ~93.0        | Baseline                     |
| Random Forest       | ~98.2        | High accuracy, fast training |
| XGBoost             | ~98.4        | Slightly better, slower      |
| LightGBM            | ~98.3        | Fast training and prediction |
---

### **Qualitative Analysis**

- All tree-based ensemble models performed exceptionally well on this dataset.
- No significant class imbalance issues were observed.
- Feature importances suggested **ECG, EMG, and GSR** as strong predictors of stress.

### **Quantitative Analysis**

- **Accuracy consistently above 98%** with ensemble methods.
- **F1-scores above 0.98** indicate robust detection without sacrificing precision or recall.
- Minimal overfitting was observed due to proper scaling and outlier handling.

---

## Final Conclusion

The stress detection models achieved **high accuracy with consistent results across different ensemble methods**. Random Forest, XGBoost, and LightGBM were all viable with negligible differences in this dataset. For practical deployment, **LightGBM or Random Forest** can be preferred due to their speed.

---

## How to Run

1. Install required libraries as per notebooks (`scikit-learn`, `xgboost`, `lightgbm`, `pandas`, `numpy`, `matplotlib`, `seaborn`).
2. `Stress_detection_model.ipynb` reads `dataset.csv` in this folder (480,052 rows; columns BVP, EDA, ACC_x, ACC_y, ACC_z, TEMP, Label). That file is the preprocessed subset from the Drive link in the notebook.
3. `Stress_detection_EDA.ipynb` still reads the larger Colab file `10000_subj_synthetic_DGAN.csv`, which is not in this folder. Its saved plots are already in the notebook.
