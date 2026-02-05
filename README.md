# Explainable Machine Learning for Credit Risk Prediction using SHAP

## 📌 Project Overview
This project focuses on building transparent and fair credit risk models for digital lending using **XGBoost** and **CatBoost**. Unlike traditional "black box" machine learning models, this study prioritizes explainability by integrating **SHAP (SHapley Additive exPlanations)** analysis. 

The goal is to predict loan default risk using borrower-oriented features (such as Debt-to-Income ratio, income, and credit history) while ensuring the model adheres to economic intuition and remains free of bias.

## 🚀 Key Features
* **Data Cleaning & Preprocessing**: Rigorous removal of data leakage features (e.g., future repayment info) and non-informative IDs. Handling of high-ratio missing values.
* **Exploratory Data Analysis (EDA)**: Numeric and categorical analysis to understand feature distributions (e.g., Annual Income, Home Ownership).
* **Advanced Modeling**: Implementation of gradient boosting algorithms:
    * **XGBoost**
    * **CatBoost**
    * *Baseline comparison with Random Forest*
* **Model Explainability**: Utilization of **SHAP** values to interpret model decisions globally and locally, identifying key drivers of credit risk.
* **Fairness Analysis**: Verification that the model learns real financial risk rather than perpetuating demographic bias.

## 🛠️ Technologies Used
* **Python 3**
* **Data Manipulation**: `pandas`, `numpy`
* **Visualization**: `matplotlib`
* **Machine Learning**: `scikit-learn`, `xgboost`, `catboost`
* **Explainability**: `shap`

## 📂 Dataset
The project utilizes a sample dataset from **LendingClub** (`lendingclub_sample_100k_raw.csv`), containing approximately 100,000 loan records with over 150 features. 
* **Target Variable**: Loan Status (Fully Paid vs. Charged Off)
* **Input Features**: DTI, Annual Income, Credit History, etc.

## ⚙️ Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/hvp007/credit-risk-prediction.git](https://github.com/hvp007/credit-risk-prediction.git)
    cd credit-risk-prediction
    ```

2.  **Install dependencies:**
    It is recommended to run this project in a virtual environment or Google Colab.
    ```bash
    pip install pandas numpy matplotlib scikit-learn xgboost catboost shap
    ```

3.  **Run the Notebook:**
    Open `Codefile.ipynb` in Jupyter Notebook or Google Colab and execute the cells sequentially.
    * *Note: If running locally, ensure the dataset path in "Step 1" matches your local directory.*

## 📊 Key Findings & Lessons
* **Predictive Power**: Borrower-centric variables (DTI, Income, Credit History) are sufficient to model credit risk effectively without needing obscure or proprietary data.
* **Model Consistency**: Both XGBoost and CatBoost provided consistent high-confidence indicators for risk evaluation.
* **Economic Intuition**: SHAP analysis confirmed the model behaves logically—risk increases with higher interest rates and DTI, and decreases with higher income and better credit history.
* **Fairness**: The models reflect the actual distribution of financial risk across borrower groups without introducing artificial bias.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome!

## 📜 License
[MIT](https://choosealicense.com/licenses/mit/)
