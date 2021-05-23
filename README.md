# Implementing the Customer Churn Analysis in Telco Industry to improving Customer retention using Pyspark in Databricks Analytics - Milestone Report
***
## About This Kernel
***

*This kernel is a part of my journey learning pyspark and how databricks can be useful for Deploying Machine Learning Algorithms in the Cloud.I use [IBM dataset](https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113) for exploring the dataset, doing exploratory data analysis and implement Machine learning algorithms to derive actionable insights. If there are any suggestions/changes you would like to see in the Kernel please let me know :). Appreciate every ounce of help!

*This notebook will always be a work in progress. Please leave any comments about further improvements to the notebook! Any feedback or constructive criticism is greatly appreciated!.

## Business Problem
***
*Telco Company wants to know whether the customer will be churn or not that contribute to the retention of Customers in using the service of the company.*

## Client
***
*Customers of telco company using IBM Dataset*

## Objective
***
*Customers are one of the most important factors in the growth of the company. The knowledge of customer's needs is essential to make a profitable and sustainable company. Hence, The process to analyze customer needs retention plays a significant part in improving the company's growth, protects loyal customers, and improve its customer relationship management (CRM). We will build an end-to-end Machine Learning Project to analyze the telco industry based on the fictional dataset provided by IBM by using pyspark in the free community edition of Databricks. This Telco churn dataset contains a fictional telco company that provided home phone and internet services to 7034 customers in California. The dataset consists of information about the customers who left, stayed, and sign up within the last month, demographic info about the customers' age range, gender, and customer account information that we will see in through Exploratory Data Analysis. My goal is to understand what factors contribute most to customers who likely to Churn and create a model that can predict if a certain customers will use the service or not(Churn).*



#### -- Project Status: [Active]
<img src="https://static1.squarespace.com/static/5144a1bde4b033f38036b7b9/t/56ab72ebbe7b96fafe9303f5/1454076676264/"/>
## Project Intro/Objective</br>
Customers are one of the most important factors in the growth of the company. The knowledge of customer's needs is essential to make a profitable and sustainable company. Hence, The process to analyze customer needs retention plays a significant part in improving the company's growth, protects loyal customers, and improve its customer relationship management (CRM). We will build an end-to-end Machine Learning Project to analyze the telco industry based on the fictional dataset provided by IBM by using pyspark in the free community edition of Databricks. This Telco churn dataset contains a fictional telco company that provided home phone and internet services to 7034 customers in California. The dataset consists of information about the customers who left, stayed, and sign up within the last month, demographic info about the customers' age range, gender, and customer account information that we will see in through Exploratory Data Analysis.

### Methods Used
* Descriptive Statistics
* Inferential Statistics
* Machine Learning Algorithms 
* Data Visualization
* Predictive Modeling

### Technologies
* Python
* PostgreSQL
* Pandas, matplotlib
* HTML
* Pyspark

## OSEMN Pipeline
****

*I’ll be following a typical data science pipeline, which is call “OSEMN” (pronounced awesome).*

1. **Obtaining** the data is the first step in solving the problem.

2. **Scrubbing** or cleaning the data is the next step. This includes data imputation of missing ovalues or improving features/columns.

3. **Exploring** the data will follow to Look for any outliers. Understanding the relationship each explanatory variable has with the response variable resides here and we can do this with a correlation matrix. 

4. **Modeling** the data will give us our predictive power on whether a customer likely to Churn or not. 

5. **Interpreting** the data is last. With all the results and analysis of the data, What are the conclusions?What factors that contribute the most about customers who likely to Churn?

**Note:** *The data was found from the “IBM Business Analytics Community” dataset provided by IBM’s website. https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113 *

**Note:** THIS DATASET IS **SIMULATED**.

# Part 1: Obtaining the Data 
***

```python 
# File location and type
file_location = "/FileStore/shared_uploads/churn_analysis/Telco_Customer_Churn.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

telco_churn = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .option('nanValue', ' ')\
  .option('nullValue', ' ')\
  .load(file_location)
```


# Part 2: Scrubbing the Data 
***

*Typically, cleaning the data requires a lot of work and can be a very tedious procedure. This dataset from IBM dataset contains a few missing values. I will have to examine the dataset to make sure that everything else is readable and that the observation values match the feature names appropriately.*

```python
# Check to see if there are any missing values in our data set
from pyspark.sql.functions import isnan,when,count,col
telco_churn.select([count(when(isnan(column) | col(column).isNull(),column)).alias(column) for column in telco_churn.columns]).show()
```

```python
# Get a quick overview of what we are dealing with in our dataset
telco_churn_df=telco_churn.toPandas()
telco_churn_df.head()
```

<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>customerID</th>
      <th>gender</th>
      <th>SeniorCitizen</th>
      <th>Partner</th>
      <th>Dependents</th>
      <th>tenure</th>
      <th>PhoneService</th>
      <th>MultipleLines</th>
      <th>InternetService</th>
      <th>OnlineBackup</th>
      <th>DeviceProtection</th>
      <th>TechSupport</th>
      <th>StreamingTV</th>
      <th>StreamingMovies</th>
      <th>PaperlessBilling</th>
      <th>PaymentMethod</th>
      <th>MonthlyCharges</th>
      <th>PaperlessBilling</th>
      <th>TotalCharges</th>
      <th>Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7590-VHVEG</th>
      <td>Female</td>
      <td>0</td>
      <td>Yes</td>
      <td>No</td>
      <td>1</td>
      <td>No</td>
      <td>No phone service</td>
      <td>DSL</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>29.85</td>
      <td>No</td>
    </tr>
    <tr>
     <th>5575-GNVDE</th>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>34</td>
      <td>Yes</td>
      <td>No</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>One year</td>
      <td>No</td>
      <td>Mailed Check</td>
      <td>56.95</td>
      <td>No</td>
    </tr>
    <tr>
     <th>5575-GNVDE</th>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>2</td>
      <td>Yes</td>
      <td>No</td>
      <td>DSL</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Mailed Check</td>
      <td>53.85</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
</div>

# Part 3: Exploring the Data
*** 
<img  src="https://s-media-cache-ak0.pinimg.com/originals/32/ef/23/32ef2383a36df04a065b909ee0ac8688.gif"/>

# Comparison between Tenure and Total Charges
```python
#Comparison between tenure and Total Charges
import matplotlib.pyplot as plt
plt.clf()
plt.plot(telco_churn_df["tenure"],telco_churn_df["TotalCharges"],".")
plt.xlabel("Tenure")
plt.ylabel("Total Charges")
plt.suptitle("The comparison between Tenure and Total Charges")
display()
```
![Comparison between Tenure and Total Charges](https://github.com/naiborhujosua/Telco_Churn_Analysis/blob/main/output0.jpeg)


# Comparison between Senior Citizen and Churn
```python
#Create temp table for exploring pyspark sql
temp_table_churn ="churn_analysis"
telco_churn.createOrReplaceTempView(temp_table_churn)

#query the data to know the correlation of SeniorCitizwen over churn
SELECT SeniorCitizen,churn,COUNT(*) FROM churn_analysis GROUP BY SeniorCitizen,churn
```
![Comparison between Senior Citizen and Churn](https://github.com/naiborhujosua/Telco_Churn_Analysis/blob/main/output1.png)



# Comparison between Churn and gender 
```python
#query the data to know the correlation of SeniorCitizwen over churn
SELECT gender,churn,COUNT(*) FROM churn_analysis GROUP BY gender, churn
```
![Comparison between Churn and gender ](https://github.com/naiborhujosua/Telco_Churn_Analysis/blob/main/output2.png)


# Comparison between tenure and churn

```python 
SELECT CAST(tenure AS int), churn, COUNT(churn) FROM churn_analysis GROUP BY tenure, churn ORDER BY CAST(tenure AS int)
```
![Comparison between tenure and churn](https://github.com/naiborhujosua/Telco_Churn_Analysis/blob/main/output3.png)<br>


# Comparison between Churn and PaymentMethod 
<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>PaymentMethod</th>
      <th>No</th>
      <th>Yes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Credit card(automatic)</th>
      <td>1290</td>
      <td>232</td>
    </tr>
    <tr>
     <th>Bank Transfer(automatic)</th>
      <td>1286</td>
      <td>258</td>
    </tr>
    <tr>
     <th>Mailed Check</th>
      <td>1304</td>
      <td>308</td>
    </tr>
    <tr>
     <th>Electronic Check</th>
      <td>1294</td>
      <td>308</td>
    </tr>
  </tbody>
</table>
</div>

##  Statistical Overview 
***

```python
telco_churn_df.printSchema()
```
The dataset has:
 - About 7043 observations and 21 features 
 - The company had categorical and numerical features
 
# 4. Modeling the Data
***
 The best model performance out of the four (Decision Tree Model, Gradient Boosting Model, Logistic Regression Model, Random Forest Model) was **Logistic Regression**! 
 
```python 
#Split the data into test and train
#We can see there are 4885 rows for train data and 2158 rows for test data
(train_data,test_data) =telco_churn.randomSplit([0.7,0.3],50)

print("Number of records for training data : {}".format(train_data.count()) + " rows")
print("Number of records for test data : {}".format(test_data.count()) + " rows")
```
 # 4a. Building Pipeline
***

```python
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer,VectorAssembler

catColumns_df = ['gender',
 'SeniorCitizen',
 'Partner',
 'Dependents',
 'PhoneService',
 'MultipleLines',
 'InternetService',
 'OnlineSecurity',
 'OnlineBackup',
 'DeviceProtection',
 'TechSupport',
 'StreamingTV',
 'StreamingMovies',
 'Contract',
 'PaperlessBilling',
 'PaymentMethod']
```

```python 
stages = []
for catColumn in catColumns_df:
  stringIndexer =StringIndexer(inputCol =catColumn,outputCol =catColumn + "Index")
  encoder  = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()],outputCols =[catColumn + "catVec"])
  
  stages += [stringIndexer,encoder]
stages

#imputing null values in totalcharges using Imputer
from pyspark.ml.feature import Imputer 
imputer = Imputer(inputCols =["TotalCharges"],outputCols =["Out_TotalCharges"])
stages += [imputer]
```

 # 4b. Create Numerical Churn column for prediction
***
```python
label_Idx = StringIndexer(inputCol="Churn",outputCol="label")
stages +=[label_Idx]
temp = label_Idx.fit(train_data).transform(train_data)
  
```

 # 4c. Make the buckets for tenure
***
```python
from pyspark.ml.feature import QuantileDiscretizer
tenure_bin =QuantileDiscretizer(numBuckets=3,inputCol="tenure",outputCol="tenure_bin")
stages +=[tenure_bin]
```

 # 4c. Join features into vector form
***
```python
numericCols =["tenure_bin","Out_TotalCharges","MonthlyCharges"]
assembleInputs =assemblerInputs =[column + "catVec" for column in catColumns_df] +numericCols
assembler = VectorAssembler(inputCols=assembleInputs,outputCol="features")
stages +=[assembler]
```
 # 4d. Feature Scalling using standarization  
***
```python
pipeline =Pipeline().setStages(stages)
pipelineModel =pipeline.fit(train_data)
trainDF =pipelineModel.transform(train_data)
testDF =pipelineModel.transform(test_data)
```
 # 4e. Implement Machine Learning Algorithms 
***
```python
#Implement Machine learning algorithm for classification
from pyspark.ml.classification import LogisticRegression

#create instance of LogisticRegression
lr =LogisticRegression(labelCol="label",featuresCol="features",maxIter=10)

#Train model with training data
lrModel =lr.fit(trainDF)
```
```python
#Check the coefficient and ntercept for the logsticregression model
print("Coefficients: {}".format(lrModel.coefficients))
print("Intercept: {}".format(lrModel.intercept))

summary = lrModel.summary
```

 # 4f. Evaluate the performance of ML Algorithms
***

```python
accuracy = summary.accuracy
falsePositiveRate = summary.weightedFalsePositiveRate
truePositiveRate = summary.weightedTruePositiveRate
fMeasure = summary.weightedFMeasure()
precision = summary.weightedPrecision
recall = summary.weightedRecall
auroc =summary.areaUnderROC
print("Accuracy: {}n\nFPR: {}\nTPR:{}\nF-measure: {}\nPrecision: {}\nRecall: {}\nAreaUnderROC: {}".format(accuracy, falsePositiveRate, truePositiveRate, fMeasure, precision, recall,auroc))
```

## Logistic Regression V.S. Random Forest V.S. Decision Tree V.S. AdaBoost Model
***

# Interpretation
***
<img src="http://www.goldbeck.com/hrblog/wp-content/uploads/2015/11/giphy-3.gif"/>


