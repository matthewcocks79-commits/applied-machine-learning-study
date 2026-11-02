\# Machine Learning with Python — Revision Notes



\## Project 01: Credit Card Fraud Detection



\### 1. Understanding the dataset



The objective was to train a machine-learning model to distinguish between legitimate and fraudulent credit-card transactions.



The dataset contained \*\*284,807 transactions\*\*, of which only \*\*492 were fraudulent\*\*.



This is an example of \*binary classification\* because there are two possible outcomes:



\- `Class = 0`: Legitimate transaction

\- `Class = 1`: Fraudulent transaction



The dataset is highly imbalanced, with fraudulent transactions accounting for approximately 0.173% of the observations.



\### 2. Exploring data using pandas



The pandas library provides tools for organising and analysing tabular data.



A \*\*DataFrame\*\* is a two-dimensional structure containing rows and columns.



Important commands:



| Command | Purpose |

|---|---|

| `df.head()` | Display the first five rows |

| `df.info()` | Examine columns, data types and missing values |

| `df.describe()` | Calculate descriptive statistics |

| `df.shape` | Return the number of rows and columns |

| `df\["Class"].value\_counts()` | Count observations in each class |

| `df.isnull().sum()` | Count missing values in each column |



\### 3. Principal Component Analysis (PCA)



PCA is a mathematical technique that transforms features into new variables called \*principal components\*.



The credit-card dataset contains 28 PCA-derived variables labelled `V1` to `V28`.



These variables are numerical representations of the original transaction characteristics.



The original features are not available, making individual predictions difficult to interpret.



\### 4. Preparing the data



\*\*Features\*\* are the variables used to make predictions.



The \*\*target\*\* is the outcome we want to predict.



For this project:



\- `X` contained the input features.

\- `y` contained the fraud classification.



The data was divided into training and test sets.



The training set was used to fit the model, while the test set was used to evaluate predictions on observations not used for training.



\### 5. Feature scaling



We used `StandardScaler` to standardise the `Time` and `Amount` features.



The transformation is:



\\\[

z=\\frac{x-\\mu}{\\sigma}

\\]



where:



\- \\(x\\) is the original value.

\- \\(\\mu\\) is the training mean.

\- \\(\\sigma\\) is the training standard deviation.



Standardisation produces features with a mean close to zero and a standard deviation close to one in the training data.



The scaler must be fitted to training data only to prevent data leakage.



\### 6. Logistic Regression



Logistic Regression is a supervised classification algorithm.



It combines input features using a linear equation:



\\\[

z=b\_0+b\_1x\_1+\\cdots+b\_nx\_n

\\]



The sigmoid function converts the result into an estimated probability:



\\\[

P(y=1)=\\frac{1}{1+e^{-z}}

\\]



The default decision threshold is 0.5.



A transaction with an estimated fraud probability of at least 0.5 is classified as fraudulent.



\### 7. Training and prediction



Three important scikit-learn methods:



| Method | Purpose |

|---|---|

| `.fit(X\_train, y\_train)` | Train the model |

| `.predict(X\_test)` | Predict classes |

| `.predict\_proba(X\_test)` | Estimate class probabilities |



\### 8. Evaluating the model



A confusion matrix identifies four possible outcomes:



| | Predicted legitimate | Predicted fraud |

|---|---|---|

| Actual legitimate | True negative | False positive |

| Actual fraud | False negative | True positive |



\*\*Precision\*\* measures the proportion of predicted fraud cases that were genuinely fraudulent.



\\\[

Precision=\\frac{TP}{TP+FP}

\\]



\*\*Recall\*\* measures the proportion of actual fraud cases detected.



\\\[

Recall=\\frac{TP}{TP+FN}

\\]



\*\*F1 score\*\* is the harmonic mean of precision and recall.



\\\[

F1=2\\frac{Precision\\times Recall}{Precision+Recall}

\\]



\### 9. Classification thresholds



Reducing the classification threshold generally increases recall but may reduce precision.



This happens because more transactions are classified as potentially fraudulent.



Changing the threshold does not retrain the model.



\### 10. Cross-validation



Cross-validation evaluates a model using several training and validation splits.



We used five-fold stratified cross-validation to compare Logistic Regression models.



Stratification maintains approximately the same proportion of fraudulent and legitimate transactions in each fold.



Preprocessing must be fitted within each training fold to prevent data leakage.



\### 11. Comparing original and balanced models



| Metric | Original | Balanced |

|---|---:|---:|

| Precision | 0.8752 | 0.0629 |

| Recall | 0.6421 | 0.9138 |

| F1 score | 0.7396 | 0.1177 |

| Average Precision | 0.7622 | 0.7568 |



The balanced model detected a greater proportion of fraudulent transactions but generated substantially more false positives.



The original model achieved the higher F1 score and was selected.



\### 12. Final model results



The selected original model produced the following confusion matrix:



| | Predicted legitimate | Predicted fraud |

|---|---:|---:|

| Actual legitimate | 56,851 | 13 |

| Actual fraud | 35 | 63 |



The model correctly identified 63 of the 98 fraudulent transactions in the test dataset.



Its fraud recall was approximately 64.29%, while its fraud precision was approximately 82.89%.



The final reported average precision was 0.7432.



Because the test set had previously been examined during threshold experiments, this result should not be treated as a fully independent final evaluation.



\### 13. Main lessons



Machine-learning accuracy can be misleading when the dataset is highly imbalanced.



Precision and recall describe different aspects of classification performance.



Feature scaling helps prepare numerical inputs for model training.



Cross-validation provides a more reliable basis for comparing models than a single training/validation split.



Class weighting does not automatically produce a better classifier.



The most appropriate model depends on the consequences of false positives and false negatives.



\---



\*These notes document my independent learning and experimentation with Python and machine learning.\*

