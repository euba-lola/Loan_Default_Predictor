
# Loan Default Predictor 💳

##  Project Overview  
In today’s financial ecosystem, accurately predicting **loan default risk** is crucial for financial institutions. Traditional methods often fail to capture the complexity of borrower behavior, which is influenced by **demographic, financial, and loan performance variables**.  

This project builds a **machine learning pipeline** that predicts the likelihood of loan default by combining **demographics, loan performance history, and previous loan records**. The end goal was to design a **Streamlit application** that allows financial institutions to easily input client details and receive real-time risk predictions.  

---

##  Dataset Description  
Three datasets were provided and merged for this project:

1. **Performance dataset** – loan repayment details per client.  
2. **Demographics dataset** – client profile (age, education, employment, bank details, etc.).  
3. **Previous loans dataset** – history of loans previously taken, repayment behaviors, interest burdens, etc.  

### Key Target Variable  
- **`flag`** → Binary outcome variable  
  - `1` → Default  
  - `0` → Non-default  

---

##  Data Preprocessing  

### 1. **Data Cleaning**  
- Handled missing values:  
  - Numerical → imputed with mean/median.  
  - Categorical → imputed with `"Unknown"`.  
- Corrected inconsistent values.  
- Standardized categorical labels.  

### 2. **Feature Engineering**  
New variables were created to improve predictive power:  
- **Ratios**:  
  - `loan_to_income_ratio = loanamount / estimated_income`  
  - `loan_to_term_ratio = loanamount / termdays`  
- **Previous Loan Aggregates**:  
  - Total number of loans per client.  
  - Average and maximum interest amounts.  
  - Average and maximum repayment amounts.  
  - Credit score statistics (avg, max).  
- **Date Features**:  
  - Extracted weekday and weekend indicators from approval dates.  

### 3. **Encoding & Scaling**  
- **Categorical variables** → OneHotEncoding.  
- **Numerical variables** → StandardScaler.  

### 4. **Outlier Detection & Handling**  
- Outliers in credit score, loan amounts, and repayment were checked with boxplots.  
- Applied capping and log transformations where appropriate.  

---

##  Exploratory Data Analysis (EDA)  

- **Distribution plots** → loan amounts, credit scores, repayment delays.  
- **Correlation analysis** → identified strongest predictors of default.  
- **Class imbalance check** → dataset was imbalanced (more non-defaulters).  
- Applied **SMOTE** to balance classes.  

---

## Model Building  

### Models Tested  
- **Logistic Regression**  
- **Random Forest Classifier**  
- **XGBoost Classifier**  
- **Neural Network (basic MLP)**  

### Evaluation Metrics  
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- ROC-AUC  
- Precision-Recall Curves  

---

##  Model Tuning & Selection  
- Performed **hyperparameter tuning** using GridSearchCV.  
- Adjusted **classification thresholds** to improve Recall/F1.  
- Compared models across metrics.  

### Final Model: **Logistic Regression**  
- Achieved the best **generalization balance** (high recall with stable precision).  
- Easy to interpret for financial institutions.  
- Saved using **joblib** for deployment.  

---

## Deployment  

The final model was deployed using **Streamlit**:  

**Live App:** [https://euba-loan-default-predictor.streamlit.app/](https://euba-loan-default-predictor.streamlit.app/)

### Features of the App  
- User inputs borrower details.  
- Model predicts **risk of default (Yes/No)**.  
- Displays **prediction probability**.  
