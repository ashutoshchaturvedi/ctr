## CTR _(Click-Through Rate)_
**CSC 591/791 Algorithms for Data Guided Business Intelligence**

_Team 10 Members: Ashutosh Chaturvedi, Harshdeep Kaur, Sameer Sharma, Surbhi Gupta, Vipul Kashyap_

> _Click-through rate (CTR)_ is the ratio of users who click on a specific link to the number of total users who view a page, email, or advertisement. It is commonly used to measure the success of an online advertising campaign for a particular website as well as the effectiveness of email campaigns.

### Dataset
Data is available at Kaggle Avazu-CTR-Prediction page: [Data Files](https://www.kaggle.com/c/avazu-ctr-prediction/data)

__File descriptions__
* train: Training set. 10 days of click-through data, ordered chronologically. Non-clicks and clicks are subsampled according to different strategies.
* test: Test set. 1 day of ads to for testing your model predictions. 

### Algorithms Implemented
* Naive Bayes
* Logistic Regression
* FTRL (Follow the Regularized Leader)

### Files:
_Logistic Regression_
* base_model.py - Files contains the code for running some base models on the given data.
* generate_data_file.py - Contains code to create smaller data files with lesser records randomly chosen from the given data set.
