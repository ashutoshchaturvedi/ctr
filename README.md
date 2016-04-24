## CTR _(Click-Through Rate)_
**CSC 591/791 Algorithms for Data Guided Business Intelligence**

_Team 10 Members: Ashutosh Chaturvedi, Harshdeep Kaur, Sameer Sharma, Surbhi Gupta, Vipul Kashyap_

> _Click-through rate (CTR)_ is the ratio of users who click on a specific link to the number of total users who view a page, email, or advertisement. It is commonly used to measure the success of an online advertising campaign for a particular website as well as the effectiveness of email campaigns.

### Files:
* base_model.py - Files contains the code for running some base models on the given data.
* generate_data_file.py - Contains code to create smaller data files with lesser records randomly chosen from the given data set.

### Command to run: 
```
python base_model.py
```

Sample Output:

SameersMac:CTR$ python base_model.py
Preprocessing the data...
Length of train data:  670000
Length of validation data:  330000
Preprocessing done for the data.

Naive Bayes :

Log Loss : 0.48859290292

Logistic Regression (Vanilla): 

Log Loss : 0.455320570096
 
Logistic Model with SGD and L2 regularization :

Log Loss : 0.455691452922
