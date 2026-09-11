# Customer_Churn_Prediction_ML
End-to-end Customer Churn Prediction ML project with data analysis, machine learning pipeline, and FastAPI REST API deployment.

The Project predicts whether a telecom customer is likely to churn based on customer demographics, service usage, contract information, and billing details.

The project covers the complete ML lifecycle — **data preprocessing → exploratory data analysis → feature engineering → model training → evaluation → model serialization → REST API development → deployment**.

---

## 🚀 Project Overview

Customer churn is a major challenge for telecom companies. Identifying customers who are likely to leave allows businesses to take proactive retention actions.

This project uses the **IBM Telco Customer Churn dataset** to build a machine learning system that:

* Analyzes customer behavior and service information
* Preprocesses numerical and categorical data
* Trains multiple machine learning models
* Evaluates model performance
* Predicts customer churn probability
* Categorizes customers into **Low, Medium, or High Risk**
* Exposes predictions through a **FastAPI REST API**

---

## 🎯 Objectives

* Predict whether a customer is likely to churn.
* Identify important factors associated with customer churn.
* Provide churn probability rather than only a binary prediction.
* Build a reusable ML pipeline containing preprocessing and the trained model.
* Deploy the trained model as a REST API.
* Create a production-oriented ML project structure suitable for further deployment.

---

## 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │   Customer Dataset  │
                    │   Telco Churn Data  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Data Cleaning     │
                    │ Missing Values      │
                    │ Data Type Handling  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │        EDA          │
                    │ Trends & Patterns   │
                    │ Churn Analysis      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Feature Processing  │
                    │ Encoding + Scaling  │
                    └──────────┬──────────┘
                               │
                               ▼
              ┌──────────────────────────────────┐
              │       Machine Learning Models    │
              │                                  │
              │ Logistic Regression              │
              │ Decision Tree                    │
              │ Random Forest                    │
              │ XGBoost                          │
              └────────────────┬─────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Model Evaluation    │
                    │ Accuracy            │
                    │ Precision           │
                    │ Recall              │
                    │ F1 Score            │
                    │ ROC-AUC             │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Trained ML Pipeline │
                    │      Joblib         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     FastAPI API     │
                    │     /predict        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Prediction Response │
                    │ Churn Prediction    │
                    │ Probability         │
                    │ Risk Level          │
                    └─────────────────────┘
```

---

## 📂 Project Structure

```text
customer-churn-prediction-ml/
│
├── api/
│   ├── __init__.py
│   └── main.py
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── model/
│   └── customer_churn_model.pkl
│
├── notebooks/
│   └── customer_churn_analysis.ipynb
│
├── screenshots/
│   ├── eda.png
│   ├── model-performance.png
│   ├── swagger.png
│   └── prediction.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

> **Note:** Avoid committing large datasets or sensitive information to GitHub. If the dataset has redistribution restrictions, provide instructions for obtaining it instead of uploading it.

---

# 📊 Dataset

The project uses the **Telco Customer Churn dataset**.

The dataset contains customer information such as:

* Customer demographics
* Tenure
* Phone services
* Internet services
* Online security
* Online backup
* Device protection
* Technical support
* Streaming services
* Contract type
* Payment method
* Monthly charges
* Total charges
* Churn status

### Target Variable

```text
Churn
```

| Value | Meaning                 |
| ----- | ----------------------- |
| 0     | Customer does not churn |
| 1     | Customer churns         |

---

# 🔄 Machine Learning Pipeline

## 1. Data Loading

The dataset is loaded using Pandas.

```python
import pandas as pd

df = pd.read_csv(
    "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)
```

---

## 2. Data Cleaning

The preprocessing stage includes:

* Removing unnecessary columns
* Handling missing values
* Converting `TotalCharges` to numeric
* Removing invalid target records
* Cleaning categorical values
* Converting the target variable into binary format

Example:

```python
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})
```

---

## 3. Exploratory Data Analysis

EDA was performed to understand:

* Churn distribution
* Customer tenure
* Monthly charges
* Contract types
* Internet services
* Payment methods
* Relationships between customer attributes and churn

Visualization libraries:

```text
Matplotlib
Seaborn
```

---

## 4. Feature Engineering & Preprocessing

The ML pipeline automatically handles numerical and categorical features.

### Numerical Features

* Missing value imputation
* Standard scaling

### Categorical Features

* Missing value imputation
* One-hot encoding
* Unknown-category handling

Implemented using:

```python
ColumnTransformer
Pipeline
StandardScaler
OneHotEncoder
SimpleImputer
```

This ensures that the **same preprocessing used during training is automatically applied during API prediction**.

---

# 🤖 Machine Learning Models

Multiple classification algorithms were considered:

| Model               | Type                      |
| ------------------- | ------------------------- |
| Logistic Regression | Linear Classification     |
| Decision Tree       | Tree-Based Classification |
| Random Forest       | Ensemble Learning         |
| XGBoost             | Gradient Boosting         |

The final model is selected based on evaluation metrics, with particular attention to **ROC-AUC, precision, recall, and F1-score** rather than accuracy alone.

---

# 📈 Model Performance

Update the following table with the **actual results from your final notebook** before publishing.

| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -------: | ------: |
| Logistic Regression |   XX.XX% |    XX.XX% | XX.XX% |   XX.XX% |   XX.XX |
| Decision Tree       |   XX.XX% |    XX.XX% | XX.XX% |   XX.XX% |   XX.XX |
| Random Forest       |   XX.XX% |    XX.XX% | XX.XX% |   XX.XX% |   XX.XX |
| XGBoost             |   XX.XX% |    XX.XX% | XX.XX% |   XX.XX% |   XX.XX |

### Evaluation Metrics

**Accuracy**
Measures the percentage of correctly classified customers.

**Precision**
Measures how many customers predicted as churners actually churned.

**Recall**
Measures how many actual churners were successfully identified.

**F1 Score**
Provides a balance between precision and recall.

**ROC-AUC**
Measures the model's ability to distinguish between churn and non-churn customers across classification thresholds.

---

# 💾 Model Serialization

The complete trained pipeline is saved using Joblib.

```python
import joblib

joblib.dump(
    best_model,
    "customer_churn_model.pkl"
)
```

The saved pipeline contains both:

```text
Preprocessing
      +
Trained ML Model
```

Therefore, the FastAPI application does not need to retrain the model for every prediction.

---

# ⚡ FastAPI Deployment

The trained model is exposed through a REST API using **FastAPI**.

## API Endpoints

### GET `/`

Checks whether the API is running.

Example response:

```json
{
    "message": "Customer Churn Prediction API is running"
}
```

---

### POST `/predict`

Accepts customer information and returns a churn prediction.

### Example Request

```json
{
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 60.0,
    "TotalCharges": 720.0
}
```

### Example Response

```json
{
    "churn_prediction": 1,
    "churn_probability": 0.7241,
    "risk_level": "High"
}
```

### Risk Classification

```text
Probability < 0.30
        ↓
      Low

0.30 - 0.59
        ↓
     Medium

Probability >= 0.60
        ↓
      High
```

---

# 📚 API Documentation

FastAPI automatically provides interactive API documentation.

After starting the application, open:

```text
http://127.0.0.1:8000/docs
```

The Swagger interface can be used to:

* View API endpoints
* Enter customer information
* Send prediction requests
* View JSON responses
* Test the model interactively

---

# 🛠️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction-ml.git
```

```bash
cd customer-churn-prediction-ml
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
```

Activate:

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
```

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📦 Requirements

Example `requirements.txt`:

```text
pandas
numpy
scikit-learn
matplotlib
seaborn
xgboost
joblib
fastapi
uvicorn
pydantic
```

---

# ▶️ Run the API Locally

From the project root:

```bash
uvicorn api.main:app --reload
```

The API will run at:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

---

# 🧪 Testing the API with Python

```python
import requests

customer = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 60.0,
    "TotalCharges": 720.0
}

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json=customer
)

print(response.json())
```

---

# 🖼️ Screenshots

Add screenshots from your actual project to make the repository more attractive to recruiters.

## Exploratory Data Analysis

![EDA](screenshots/eda.png)

## Model Performance

![Model Performance](screenshots/model-performance.png)

## FastAPI Swagger Documentation

![Swagger API](screenshots/swagger.png)

## Churn Prediction Response

![Prediction](screenshots/prediction.png)

> Replace these image paths with your actual screenshots.

---

# ☁️ Deployment

The FastAPI application can be deployed to a cloud platform that supports Python web services.

General deployment process:

```text
GitHub Repository
       │
       ▼
Cloud Hosting Platform
       │
       ▼
Install requirements.txt
       │
       ▼
Start Uvicorn
       │
       ▼
FastAPI Application
       │
       ▼
Public API URL
```

## Production Start Command

Use:

```bash
uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

The `$PORT` variable is provided by many cloud hosting platforms.

### Deployment Checklist

Before deploying:

* [ ] Push project to GitHub
* [ ] Add `requirements.txt`
* [ ] Add `api/__init__.py`
* [ ] Add `api/main.py`
* [ ] Add trained model
* [ ] Verify model loading
* [ ] Test `/` endpoint
* [ ] Test `/predict`
* [ ] Verify Swagger documentation
* [ ] Configure production start command
* [ ] Test the public API URL

---

# 🔐 Security & Production Considerations

For a production application, the following improvements can be added:

* API authentication
* Input validation
* Logging
* Error handling
* Rate limiting
* HTTPS
* Docker containerization
* CI/CD pipeline
* Model versioning
* Monitoring and model performance tracking

---

# 🔮 Future Improvements

* Add a web-based frontend for customer predictions
* Implement explainable AI using SHAP
* Add automated model retraining
* Add MLflow for experiment tracking
* Add Docker support
* Implement CI/CD using GitHub Actions
* Deploy the API to a cloud platform
* Add database integration for prediction history
* Build a customer-retention recommendation system
* Add model monitoring

---

# 🧠 Key Learning Outcomes

Through this project, I gained practical experience in:

* Data preprocessing
* Exploratory Data Analysis
* Feature engineering
* Classification algorithms
* Model evaluation
* Machine learning pipelines
* Hyperparameter tuning
* Model serialization
* REST API development
* FastAPI
* API testing
* Git and GitHub
* ML deployment concepts

---

# 👨‍💻 Author

**Sanisetty Preetham**

B.Tech – Computer Science & Engineering (AI & Data Science)

Reva University, Bangalore

**Areas of Interest:**
Data Analytics | Machine Learning | Python | FastAPI | AI/ML

---

# ⭐ If You Find This Project Useful

If you find this project helpful, consider giving the repository a ⭐ star.
