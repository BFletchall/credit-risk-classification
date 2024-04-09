# credit-risk-classification

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* The purpose of this analysis is to develop a machine learning model to identify the creditworthiness of borrowers.
* The financial information used is a dataset of historical lending activity from a peer-to-peer lending services company.
* The dataset includes information about the loan status and various features related to borrowers, which will be used to train and evaluate the model. Specifically, we will use logistic regression to classify loans as either healthy (0) or high-risk (1).
* The variable we are trying to predict is the "loan_status," which indicates whether a loan is healthy (0) or high-risk (1). To gain a better understanding of the distribution of loan statuses in the dataset, we used the value_counts() method, which provides the count of unique values in a column.
* Stages of the Machine Learning Process:
    - Data Preprocessing - We read the lending data from the provided CSV file and split it into features (X) and labels (y). Then, we split the data into training and testing sets using the train_test_split function.
    - Model Training: We employed logistic regression, a popular classification algorithm, to train our model. This involved fitting the logistic regression model to the training data using LogisticRegression from scikit-learn.
    - Model Evaluation: After training the model, we evaluated its performance using various metrics, including a confusion matrix, classification report, accuracy, precision, and recall scores.
* Methods Used:
- Logistic Regression: Logistic regression is a widely used statistical technique for binary classification problems, making it suitable for predicting loan status (healthy or high-risk) in our analysis. We utilized the LogisticRegression class from scikit-learn to implement this algorithm.
- train_test_split: This method from scikit-learn was used to split the dataset into training and testing sets, allowing us to train the model on one subset and evaluate its performance on another.

## Results

Using bulleted lists, describe the accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
    * Accuracy: The overall accuracy of the model is 0.99, indicating that it correctly predicts the labels for 99% of the instances in the dataset.

    * Precision: Precision measures the accuracy of positive predictions. For class 0 (healthy loan), the precision is 1.00, indicating that almost all of the instances predicted as healthy loans are actually healthy loans. For class 1 (high-risk loan), the precision is 0.85, indicating that about 85% of the instances predicted as high-risk loans are actually high-risk loans.

    * Recall: Recall measures the ability of the model to capture positive instances. For class 0 (healthy loan), the recall is 0.99, indicating that the model correctly identifies 99% of the actual healthy loans. For class 1 (high-risk loan), the recall is 0.91, indicating that the model captures 91% of the actual high-risk loans.

    * F1-score: The F1-score is the harmonic mean of precision and recall, providing a balance between the two metrics. For class 0 (healthy loan), the F1-score is 1.00, indicating excellent performance. For class 1 (high-risk loan), the F1-score is 0.88, which is slightly lower but still indicates good overall performance.

    * Support: Support indicates the number of actual occurrences of each class in the dataset. There are 18,765 instances of class 0 (healthy loan) and 619 instances of class 1 (high-risk loan).

## Summary

The model demonstrates high accuracy, precision, and recall scores, indicating its effectiveness in identifying both healthy and high-risk loans. With an accuracy of 0.99 and precision and recall scores above 0.85 for high-risk loans, the logistic regression model appears to be well-suited for practical use by the lending company.

From the provided classification report, we can analyze how well the logistic regression model predicts both the "healthy loan" (0) and "high-risk loan" (1) labels using various evaluation metrics:

Overall, the logistic regression model performs very well in predicting both the "healthy loan" and "high-risk loan" labels. It achieves high precision, recall, and F1-score for the majority class (0), and although the minority class (1) has slightly lower scores, it still demonstrates good predictive performance. 