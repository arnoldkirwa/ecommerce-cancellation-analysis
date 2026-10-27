# E-Commerce Order Cancellation Risk Analysis

![E-Commerce Order Cancellation Risk Analysis](https://github.com/arnoldkirwa/ecommerce-cancellation-analysis/raw/main/project_cover.png)

## 1. Project Overview

This project analyzes e-commerce order cancellations using the **Brazilian E-Commerce Public Dataset by Olist**. It applies data analysis and machine learning to identify patterns associated with canceled orders and evaluate whether cancellation risk can be predicted before fulfillment.

The project follows the **CRISP-DM (Cross-Industry Standard Process for Data Mining)** methodology, covering business understanding, data understanding, data preparation, modeling, and evaluation. It also includes an interactive Streamlit application for demonstrating cancellation-risk predictions.

**Project status:** Completed portfolio project with an experimental machine learning model and deployed Streamlit demonstration.

## 2. Live Demo

**Try the application:** https://ecommerce-cancellation-analysis-rhmmhqftttojpxsn5qlugq.streamlit.app/

The Streamlit application allows users to enter order characteristics and view a predicted cancellation risk score.

> **Important:** The application is a portfolio demonstration, not a production-ready cancellation prevention system. The model has low cancellation precision, so its predictions should not be treated as reliable operational alerts.

## 3. Business Problem

Order cancellations can lead to lost revenue, wasted operational effort, and poor customer experiences. E-commerce businesses need to understand cancellation patterns and identify orders that may be at higher risk.

This project investigates whether historical order information can help identify patterns associated with cancellations and support better operational decisions.

## 4. Project Objectives

- Analyze order data and identify cancellation patterns.
- Investigate relationships between order characteristics and cancellations.
- Address the severe class imbalance between canceled and delivered orders.
- Prepare an order-level analytical dataset.
- Engineer a binary cancellation target.
- Train and compare multiple classification models.
- Evaluate performance using metrics appropriate for imbalanced classification.
- Investigate important predictive features and document limitations.
- Develop and deploy an interactive Streamlit application.

## 5. Dataset

**Dataset:** Brazilian E-Commerce Public Dataset by Olist

**Source:** [Olist Brazilian E-Commerce Dataset on Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

The dataset contains information about orders, products, payments, customers, sellers, and delivery. Relevant tables were combined to create an order-level analytical dataset.

### Expanded Modeling Dataset

After excluding canceled orders with missing product features, the expanded modeling dataset contains:

| Metric | Value |
|---|---:|
| Total orders | 96,939 |
| Delivered orders | 96,478 |
| Canceled orders | 461 |
| Cancellation rate | Approximately 0.48% |

The expanded dataset excludes 164 canceled orders with missing product information. This exclusion may introduce selection bias and is an important limitation of the analysis.

## 6. Methodology — CRISP-DM

### 6.1 Business Understanding

Defined the business problem of investigating and predicting e-commerce order cancellations.

### 6.2 Data Understanding

- Inspected the structure and quality of the Olist datasets.
- Examined order statuses and cancellation prevalence.
- Investigated missing values and relevant data relationships.
- Explored patterns in order timing, product characteristics, and shipping costs.

### 6.3 Data Preparation

- Combined relevant datasets into an order-level analytical dataset.
- Engineered purchase date and time features.
- Created the binary target variable `is_canceled`.
- Prepared numerical and categorical predictors.
- Excluded identifiers and post-outcome information to reduce target leakage.
- Used a stratified train–test split to preserve the class distribution.

### 6.4 Feature Engineering

The expanded model used the following 12 features:

- `purchase_year`
- `purchase_month`
- `purchase_day_of_week`
- `purchase_hour`
- `is_weekend`
- `total_order_items`
- `total_item_price`
- `total_freight_value`
- `average_item_price`
- `distinct_product_categories`
- `total_distinct_products`
- `primary_product_category`

### 6.5 Modeling

Four classification approaches were trained and compared:

- Logistic Regression
- Decision Tree
- Random Forest
- Histogram-based Gradient Boosting

The models were evaluated in the context of a highly imbalanced classification problem.

### 6.6 Evaluation

Model performance was assessed using:

- **Average Precision:** Evaluates ranking quality for the minority class across precision–recall thresholds.
- **Precision:** Measures the proportion of predicted cancellations that were actually canceled.
- **Recall:** Measures the proportion of actual cancellations identified by the model.
- **F1-score:** Combines precision and recall.
- **Train–test comparison:** Helps identify potential overfitting.
- **Permutation importance:** Investigates which features contribute to model performance.

## 7. Model Results

The following results were obtained on the expanded experiment's test set.

| Model | Average Precision | Cancellation Precision | Cancellation Recall | Cancellation F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.0082 | 0.0074 | 0.5761 | 0.0147 |
| Decision Tree | 0.0113 | 0.0087 | 0.6739 | 0.0173 |
| Random Forest | 0.0107 | 0.0100 | 0.4565 | 0.0197 |
| **Gradient Boosting** | **0.0154** | **0.0114** | **0.6739** | **0.0224** |

### Best-Performing Expanded Model

The expanded Gradient Boosting model achieved the highest Average Precision and F1-score among the four models evaluated in this experiment.

At the default classification threshold of 0.5, the model achieved:

| Metric | Result |
|---|---:|
| Average Precision | 0.0154 |
| Cancellation Precision | 1.14% |
| Cancellation Recall | 67.39% |
| Cancellation F1-score | 0.0224 |

### Interpretation

The model identified approximately 67.39% of canceled orders in the test set. However, its cancellation precision was only 1.14%, meaning that only a small proportion of orders flagged as canceled were actually canceled.

This trade-off demonstrates why recall alone is not sufficient for evaluating a cancellation-risk model. The low precision indicates a high number of false-positive predictions, limiting the model's immediate business usefulness.

The expanded model's results are experimental and require further validation before operational use.

## 8. Key Findings

Permutation importance identified the following features among the most influential in the expanded Gradient Boosting model:

1. Purchase year
2. Total freight value
3. Primary product category
4. Average item price
5. Purchase month

These findings suggest that order timing, shipping costs, product categories, and pricing contain predictive information in this dataset.

Feature importance indicates a model's reliance on features; it does not establish that those features cause cancellations.

## 9. Business Recommendations

Based on the analysis, the following actions could be investigated:

- Analyze cancellation rates across product categories and shipping-cost ranges.
- Monitor cancellation patterns over time to identify changing trends.
- Evaluate targeted interventions for potentially high-risk orders.
- Compare the costs of false-positive alerts with the potential benefits of preventing cancellations.
- Investigate additional features that may improve cancellation-risk prediction.
- Validate the model on future orders before using predictions in business operations.

These are recommendations for further investigation, not proven interventions from this experiment.

## 10. Limitations

- **Severe class imbalance:** Canceled orders represent approximately 0.48% of the expanded modeling dataset.
- **Low precision:** Many orders flagged as potential cancellations are actually delivered.
- **Excluded canceled orders:** The expanded analysis excludes 164 canceled orders with missing product features, which may introduce selection bias.
- **Generalization:** Random train–test evaluation does not establish performance on future periods.
- **Prediction timing:** Feature availability must be verified to ensure predictions can be generated at the point when a business could act.
- **Experimental performance:** The model requires further feature development, threshold evaluation, and validation before operational use.

## 11. Tools and Technologies

- **Programming language:** Python
- **Data analysis:** Pandas, NumPy
- **Data visualization:** Matplotlib, Seaborn
- **Machine learning:** Scikit-learn
- **Model persistence:** Joblib
- **Development environment:** Jupyter Notebook, VS Code
- **Version control:** Git, GitHub
- **Deployment:** Streamlit

## 12. Repository Structure

```text
ecommerce-cancellation-analysis/
│
├── app.py
├── cancellation_prediction_pipeline.ipynb
├── cancellation_risk_model.joblib
├── project_cover.png
├── README.md
└── requirements.txt
```

- `app.py` — Streamlit application for demonstrating cancellation-risk predictions.
- `cancellation_prediction_pipeline.ipynb` — Data analysis, feature engineering, model training, evaluation, and interpretation.
- `cancellation_risk_model.joblib` — Saved trained model used by the application.
- `project_cover.png` — Project cover image.
- `README.md` — Project documentation.
- `requirements.txt` — Python dependencies.

## 13. How to Run the Project Locally

### Prerequisites

- Python installed
- Git installed
- Jupyter Notebook or VS Code

### Step 1: Clone the Repository

```bash
git clone https://github.com/arnoldkirwa/ecommerce-cancellation-analysis.git
cd ecommerce-cancellation-analysis
```

### Step 2: Create and Activate a Virtual Environment (Optional)

```bash
python -m venv venv
```

On Windows Git Bash:

```bash
source venv/Scripts/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Notebook

Open `cancellation_prediction_pipeline.ipynb` in Jupyter Notebook or VS Code.

If you want to reproduce the analysis from the raw data, download the Olist dataset and place the required files in the appropriate local data directory. Update the dataset paths in the notebook to match your environment.

### Step 5: Run the Streamlit Application

Ensure the saved model file, `cancellation_risk_model.joblib`, is in the expected project directory, then run:

```bash
streamlit run app.py
```

The application should open in your browser.

## 14. Future Improvements

Potential future work includes:

- Investigating additional features available before fulfillment.
- Examining strategies for handling severe class imbalance.
- Evaluating alternative models and probability calibration.
- Selecting classification thresholds based on business costs and benefits.
- Testing performance on a time-based holdout set.
- Investigating the impact of missing product information.
- Measuring model performance and business value before considering operational use.

## 15. Conclusion

This project demonstrates an end-to-end analytical workflow for investigating e-commerce order cancellations, from data preparation and exploratory analysis to machine learning evaluation and deployment of an interactive demonstration.

Gradient Boosting was the strongest model in the expanded experiment based on Average Precision and F1-score. However, low cancellation precision, severe class imbalance, and dataset limitations mean that its predictions should be treated as exploratory rather than production-ready.

The project provides a foundation for further research into cancellation prevention, improved predictive features, and more effective operational decision-making.

---

**Author:** Arnold Kirwa

**GitHub Repository:** https://github.com/arnoldkirwa/ecommerce-cancellation-analysis

**Live Demo:** https://ecommerce-cancellation-analysis-rhmmhqftttojpxsn5qlugq.streamlit.app/
