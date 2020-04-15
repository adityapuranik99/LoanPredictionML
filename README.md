# Loan Prediction
Bank Loan default prediction through various ML models. 

## Problem Statement 

There is a company named **Dream Housing Finance** that deals in all home loans. They have presence across
all urban, semi urban and rural areas. Customer first apply for home loan after that company validates the
customer eligibility for loan. However doing this manually takes a lot of time. Hence it wants to automate the
loan eligibility process (real time) based on customer information.So the final thing is to identify the factors/
customer segments that are eligible for taking loan. How will the company benefit if we give the customer
segments is the immediate question that arises. The solution is . . . .Banks would give loans to only those
customers that are eligible so that they can be assured of getting the money back. Hence the more accurate
we are in predicting the eligible customers the more beneficial it would be for the Dream Housing Finance
Company.

### Dataset

Refer to our toy dataset: https://www.kaggle.com/altruistdelhite04/loan-prediction-problem-dataset

## Pre-processing and Data Cleaning

#### Exploratory Data Analysis

An in-depth EDA through, uni-variate as well as bivariate analysis, was done on all the numerical, categorical and ordinal features.

#### Data Cleaning

To treat the null values, we imputed median for numerical and ordinal features and mode for our categorical features.
Outliers were kept since the data set was small. 
One hot coding using ```pd.get_dummies()``` was done on the categorical variables and a correlation matrix was plotted to get a initial idea of non-contributing columns.

### Initial Model Building
The initial model was built upon the enocoded features. **K-fold and  cross validation** were implemented to iniitially test the accuracy of the 3 models. **K-fold cross validation** is a re-sampling procedure used to evaluate machine learning models on a limited data sample. **Cross-validation** is primarily used in applied machine learning to estimate the skill of a
machine learning model on unseen data. The 3 models used to initially train our dataset were:

1. Logistic Regression
2. KNN
3. Naive Bayes

### Feature Engineering

Three new features were introduced.

1. TotalIncome: Total Income of the family applying. The TotalIncome was right skewed. As we take the log transformation, it does not affect the smaller values much, but it reduces the larger values. So we get a distribution similar to a normal distribution. After the log transformation the extreme values have been subsided. 

    > TotalIncome = ApplicantIncome + CoapplicantIncome
    
    
2. EMI: EMI to be paid each month 
    > EMI = Loan_Amount/Loan_Amount_Term
    
3. BalanceIncome : The money left in hand after paying the EMI. More is the money left, less are the chances of defaulters.

    > BalanceIncome = TotalIncome - EMI
   
To avoid large correlation, we dropped ApplicantIncome, CoapplicantIncome, Loan_Amount, Loan_Amount_Term.

## Final Model Building

Our data set is a little imbalanced. So instead of looking at accuracy to compare the models, we have went with AUROC here. Though we did look into other options like penalising the algos, and using the Decision Tree Classifier/Random Forest Classifier, we still kept our primary parameter as AUROC. Accuracy was also referred too. 

We trained our dataset on following models:

1. Logistic Regression :  We penalised the algorithm here by balancing the classes. 

> class_weights = 'balanced'

2. KNN
3. Naive Bayes
4. Support Vector Machine Gaussian : Again here, the algorithm was penalised. 
5. Decision Tree Classifier
6. Random Forest Classifier

The Random Forest Classifier performed best on this dataset. 

### Setting Decision Threshold

As such, a simple and straightforward approach to improving the performance of a classifier that predicts probabilities on an imbalanced classification problem is to tune the threshold used to map probabilities to class labels. We referred to tune the decision threshold and to find the optimal point.
The Geometric Mean or G-Mean is a metric for imbalanced classification that, if optimized, will seek a balance between the sensitivity and the specificity.

> G-Mean = sqrt(Sensitivity * Specificity)

We are testing the model with each threshold returned from the call roc_auc_score() and select the thresholdwith the largest G-Mean value.
Since, the AUROC of the Random Forest curve is maximum, we have went forward with looking for its decision threshold.

### Feature Importance

*Feature importance* gives you a score for each feature of your data, the higher the score more important or relevant is the feature towards your output variable. Here, we have used **Random Forest** to plot the the graph for feature importance. **Credit History** is the most important feature in determining the loan status of an individual, followed by **TotalIncome**, and **Loan Amount_log**. 


**Please refer to report for detailed analysis.**
