# E-Commerce Order Cancellation Risk Analysis

![E-Commerce Order Cancellation Risk Analysis](project_cover.png)

# E-Commerce Order Cancellation Risk Analysis

## 1. Project Overview

This project analyzes e-commerce order cancellations using the Brazilian E-Commerce Public Dataset by Olist. It applies data analysis and machine learning to identify patterns associated with canceled orders and evaluate whether cancellation risk can be predicted before fulfillment.

The project follows the **CRISP-DM methodology** (Cross-Industry Standard Process for Data Mining), from business understanding to model evaluation.

## 2. Business Problem

Order cancellations can result in lost revenue, wasted operational effort, and poor customer experiences. E-commerce businesses need to understand cancellation patterns and identify orders that may be at higher risk.

This project explores how historical order information can help businesses understand cancellation risk and support better operational decisions.

## 3. Project Objectives

- Analyze order data and identify cancellation patterns.
- Investigate relationships between order characteristics and cancellations.
- Address the severe class imbalance between canceled and delivered orders.
- Train and compare classification models.
- Evaluate model performance using metrics appropriate for imbalanced data.
- Develop evidence-based business recommendations.

## 4. Dataset

**Source:** Brazilian E-Commerce Public Dataset by Olist.

The dataset contains information about orders, products, payments, customers, and delivery. The project combines relevant order-level information to create a modeling dataset.

The expanded modeling dataset contains **96,939 orders**, including 461 canceled orders and 96,478 delivered orders, after excluding records with missing product features.

The cancellation rate in this expanded dataset is approximately **0.48%**.

## 5. Methodology — CRISP-DM

### Business Understanding
Defined the business problem of understanding and predicting e-commerce order cancellations.

### Data Understanding
Inspected the dataset, examined order statuses, analyzed missing values, and explored cancellation prevalence.

### Data Preparation
Combined relevant order information, engineered purchase date and time features, prepared numerical and categorical predictors, and excluded identifiers and information that could leak the target outcome.

### Modeling
Trained and compared four classification approaches:

- Logistic Regression
- Decision Tree
- Random Forest
- Histogram-based Gradient Boosting

Class imbalance was addressed using class weighting in the models.

### Evaluation
Compared models using average precision, precision, recall, F1-score, and train–test performance. Permutation importance was also used to investigate feature contributions.

### Deployment
This project focuses on analysis, model development, and evaluation. Production deployment is outside its current scope.

## 6. Model Results

The expanded models produced the following test-set results:

| Model | Average Precision | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.0082 | 0.0074 | 0.5761 | 0.0147 |
| Decision Tree | 0.0113 | 0.0087 | 0.6739 | 0.0173 |
| Random Forest | 0.0107 | 0.0100 | 0.4565 | 0.0197 |
| Gradient Boosting | **0.0154** | **0.0114** | **0.6739** | **0.0224** |

Gradient Boosting achieved the highest average precision and F1-score among the expanded models evaluated. At a classification threshold of 0.5, it detected approximately 67.39% of canceled orders in the test set.

However, its precision was only 1.14%, meaning that most orders flagged as potential cancellations were actually delivered. The model therefore requires further improvement and validation before practical use.

## 7. Key Findings

Permutation importance identified the following as the most influential features in the expanded Gradient Boosting model:

- Purchase year
- Total freight value
- Primary product category
- Average item price
- Purchase month

These results indicate that order timing, shipping cost, product category, and pricing contain useful predictive information in this dataset. Feature importance does not establish causation.

## 8. Business Recommendations

- Investigate cancellation patterns across product categories and shipping-cost ranges.
- Monitor cancellation rates over time to identify changing patterns.
- Evaluate targeted interventions for potentially high-risk orders before using predictions operationally.
- Measure the cost of false alarms against the potential benefit of preventing cancellations.
- Improve model performance and validate it on future orders before making business decisions.

## 9. Limitations

- The dataset is highly imbalanced, with canceled orders representing less than 1% of the expanded modeling data.
- Precision is low, resulting in many false-positive predictions.
- The expanded analysis excluded 164 canceled orders with missing product records. These exclusions may bias the results and limit generalization to all orders.
- Random train–test evaluation does not establish how well the model will perform on future periods.
- The features and prediction timing require further validation to ensure they are available when a business would need to act.

## 10. Tools and Technologies

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Jupyter Notebook
- Git and GitHub

## 11. How to Run the Project

1. Clone this repository.
2. Install the required Python libraries.
3. Download the Brazilian E-Commerce Public Dataset by Olist and place the files in the appropriate local data directory.
4. Update the dataset paths in the notebook to match your environment.
5. Open `cancellation_prediction_pipeline.ipynb` in Jupyter Notebook or VS Code.
6. Run the notebook cells in order.

Install the core libraries with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

## 12. Conclusion

This project demonstrates an end-to-end analytical workflow for investigating e-commerce order cancellations and evaluating machine learning approaches to cancellation-risk prediction.

Gradient Boosting was the strongest expanded model tested, but low precision and dataset limitations mean the current results should be treated as exploratory rather than production-ready.

<<<< HEAD
The analysis provides a foundation for further research into cancellation prevention, improved predictive features, and more effective operational decision-making.


