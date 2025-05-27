# Codsoft

# 💼 CodSoft Internship - Machine Learning Projects

This repository contains three major Machine Learning projects completed as part of the CodSoft internship program. Each project solves a real-world classification problem using supervised learning algorithms and Python's powerful data science ecosystem.

## 📁 Project List

### 1. 🛡️ Credit Card Fraud Detection

- **Objective**: Detect fraudulent credit card transactions based on anonymized financial data.
- **Dataset**: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Features**:
  - Preprocessed anonymized numerical features (V1-V28)
  - Imbalanced class problem (fraud vs non-fraud)
- **Model(s)**: Logistic Regression, Random Forest
- **Techniques**:
  - Data balancing using `SMOTE`
  - Model evaluation using confusion matrix, classification report, ROC AUC
- **Libraries**: `pandas`, `sklearn`, `matplotlib`, `imbalanced-learn`

### 2. 📉 Customer Churn Prediction

- **Objective**: Predict whether a customer will churn (leave) based on their activity and demographic data.
- **Dataset**: [Bank Customer Churn Prediction Dataset](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction)
- **Features**:
  - Customer demographics, bank usage behavior
  - Binary churn label
- **Model(s)**: Logistic Regression, Random Forest, Gradient Boosting
- **Techniques**:
  - Data cleaning and encoding (LabelEncoder, OneHotEncoder)
  - Train-test split and accuracy evaluation
- **Libraries**: `pandas`, `sklearn`, `matplotlib`, `seaborn`

### 3. 📩 Spam SMS Detection

- **Objective**: Classify SMS messages as spam or ham (legitimate).
- **Dataset**: [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Features**:
  - Raw SMS text messages
  - Labels: 'spam' or 'ham'
- **Model(s)**: Naive Bayes
- **Techniques**:
  - Text preprocessing
  - Feature extraction using TF-IDF
  - Model training and accuracy evaluation
- **Libraries**: `pandas`, `sklearn`
