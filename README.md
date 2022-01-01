<div align="center">
  
[1]: https://github.com/Pradnya1208
[2]: https://www.linkedin.com/in/pradnya-patil-b049161ba/
[3]: https://public.tableau.com/app/profile/pradnya.patil3254#!/
[4]: https://twitter.com/Pradnya1208


[![github](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c292abd3f9cc647a7edc0061193f1523e9c05e1f/icons/git.svg)][1]
[![linkedin](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/9f5c4a255972275ced549ea6e34ef35019166944/icons/iconmonstr-linkedin-5.svg)][2]
[![tableau](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/e257c5d6cf02f13072429935b0828525c601414f/icons/icons8-tableau-software%20(1).svg)][3]
[![twitter](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c9f9c5dc4e24eff0143b3056708d24650cbccdde/icons/iconmonstr-twitter-5.svg)][4]

</div>

# <div align="center">Walmart store sales prediction</div>
<div align="center">

<img src  ="https://github.com/Pradnya1208/Walmart-store-sales-prediction/blob/main/output/intro.png?raw=true" width="100%">
</div>


## Objectives:
Data from Walmart stores accross the US is given, and it is up to us to forecast their weekly sales. The data is already split into a training and a test set, and we want to fit a model to the training data that is able to forecast those weeks sales as accurately as possible. In fact, our metric of interest will be the Mean Absolute Error.
## Dataset:
[Walmart - Store Sales Forecasting](https://www.kaggle.com/avelinocaio/walmart-store-sales-forecasting/data)

This is the historical sales data for 45 Walmart stores located in different regions. Each store contains a number of departments
In addition, Walmart runs several promotional markdown events throughout the year. These markdowns precede prominent holidays, the four largest of which are the Super Bowl, Labor Day, Thanksgiving, and Christmas. The weeks including these holidays are weighted five times higher in the evaluation than non-holiday weeks.

**stores.csv**:
This file contains anonymized information about the 45 stores, indicating the type and size of store.

**train.csv**:

This is the historical training data, which covers to 2010-02-05 to 2012-11-01. Within this file you will find the following fields:

- Store - the store number
- Dept - the department number
- Date - the week
- Weekly_Sales -  sales for the given department in the given store
- IsHoliday - whether the week is a special holiday week

**test.csv**:

This file is identical to train.csv, except we have withheld the weekly sales. You must predict the sales for each triplet of store, department, and date in this file.

**features.csv**:

This file contains additional data related to the store, department, and regional activity for the given dates. It contains the following fields:

- Store - the store number
- Date - the week
- Temperature - average temperature in the region
- Fuel_Price - cost of fuel in the region
- MarkDown1-5 - anonymized data related to promotional markdowns that Walmart is running. MarkDown data is only available after Nov 2011, and is not available for all stores all the time. Any missing value is marked with an NA.
- CPI - the consumer price index
- Unemployment - the unemployment rate
- IsHoliday - whether the week is a special holiday week

For convenience, the four holidays fall within the following weeks in the dataset (not all holidays are in the data):

- Super Bowl: 12-Feb-10, 11-Feb-11, 10-Feb-12, 8-Feb-13
- Labor Day: 10-Sep-10, 9-Sep-11, 7-Sep-12, 6-Sep-13
- Thanksgiving: 26-Nov-10, 25-Nov-11, 23-Nov-12, 29-Nov-13
- Christmas: 31-Dec-10, 30-Dec-11, 28-Dec-12, 27-Dec-13
## Implementation:

**Libraries:** `sklearn` `Matplotlib` `pandas` `seaborn` `NumPy` 
## A few glimpses of EDA:
In this section, we will explore the datasets provided, join information between some of them and make relevant transformations.

#### Holiday Analysis:
We will analyze the week days that the Holidays fall on each year. This is relevant to know how many pre-holiday days are inside each Week marked as 'True' inside 'IsHoliday' field.

If, for a certain Week, there are more pre-holiday days in one Year than another, then it is very possible that the Year with more pre-holiday days will have greater Sales for the same Week. So, the model will not take this consideration and we might need to adjust the predicted values at the end.

Another thing to take into account is that Holiday Weeks but with few or no pre-holiday days might have lower Sales than the Week before.

We can use SQL, putting the week days for each Holiday in every year. Doing some research, the Super Bowl, Labor Day and Thanksgiving fall on the same day. In the other hand, Christmas is always on December 25th, so the week day can change.

![holidays](https://github.com/Pradnya1208/Walmart-store-sales-prediction/blob/main/output/holidays.PNG?raw=true)

** Average Weekly Sales per Year**:
![avgsales](https://github.com/Pradnya1208/Walmart-store-sales-prediction/blob/main/output/avgsales.PNG?raw=true)
As we can see, there is one important Holiday not included in 'IsHoliday'. It's the Easter Day. It is always on Sunday, but can fall on different weeks.

- In 2010 it is in Week 13
- In 2011, Week 16
- Week 14 in 2012
- Week 13 in 2013 for Test set
So, we can set the flag to 'True' for these observations.

#### Average sales per store and department:
![salesperstore](https://github.com/Pradnya1208/Walmart-store-sales-prediction/blob/main/output/storedept.PNG?raw=true)
![perdept](https://github.com/Pradnya1208/Walmart-store-sales-prediction/blob/main/output/perdept.PNG?raw=true)

#### Variable Correlation:
![corr](https://github.com/Pradnya1208/Walmart-store-sales-prediction/blob/main/output/corr.PNG?raw=true)

#### Analyzing Variables:
We will use following plots:
The **discrete plot** is for finite numbers. We will use `boxplot`, to see the medians and interquartile ranges, and the striplot, which is a better way of seeing the distribution, even more when lots of outliers are present.

The **continuous plot**, as the name says, is for continuous variables. We will see the distribution of probabilities and use `BoxCox` to understand if there is increase of correlation and decrease of skewness for each variable. In some cases the process of transforming a variable can help, depending on the model.

** Weekly Sales X IsHoliday**
![plot1](https://github.com/Pradnya1208/Walmart-store-sales-prediction/blob/main/output/plot1.PNG?raw=true)

**Weekly Sales X Type**
![plot2](https://github.com/Pradnya1208/Walmart-store-sales-prediction/blob/main/output/plot2.PNG?raw=true)

**Weekly Sales X Temperature**
![plot3](https://github.com/Pradnya1208/Walmart-store-sales-prediction/blob/main/output/plot3.PNG?raw=true)

**Weekly Sales X CPI**
![plot4](https://github.com/Pradnya1208/Walmart-store-sales-prediction/blob/main/output/plot4.PNG?raw=true)

**Weekly Sales X Unemployment**:
![plot5](https://github.com/Pradnya1208/Walmart-store-sales-prediction/blob/main/output/plot5.PNG?raw=true)

**Weekly Sales X Size**:
![plot6](https://github.com/Pradnya1208/Walmart-store-sales-prediction/blob/main/output/plot6.PNG?raw=true)

## Model Training and Evaluation:
As we can see in the figure below, the evaluation is based on Weighted Mean Absolute Error (WMAE), with a weight of 5 for Holiday Weeks and 1 otherwise.
![form](https://github.com/Pradnya1208/Walmart-store-sales-prediction/blob/main/output/form.PNG?raw=true)

```
def WMAE(dataset, real, predicted):
    weights = dataset.IsHoliday.apply(lambda x: 5 if x else 1)
    return np.round(np.sum(weights*abs(real-predicted))/(np.sum(weights)), 2)
```

The model chosen for this project is the Random Forest Regressor. It is an ensemble method and uses multiples decision trees ('n_estimators' parameter of the model) to determine final output, which is an average of the outputs of all trees.
![model](https://github.com/Pradnya1208/Walmart-store-sales-prediction/blob/main/output/model.PNG?raw=true)

The algorithm chooses a feature to be the Root Node and make a split of the samples. The function to measure the quality of a split we can choose and pass as a parameter. The splitting continues until the Internal Node has less samples than 'min_samples_split' to split and become a Leaf Node. And the 'min_samples_leaf' tells the minimum number of samples to be considered as a Leaf Node. There is also an important parameter called 'max_features' and it is the maximum number of features considered when the node is requiring the best split. The number of layers is the 'max_depth' parameter.
```
def random_forest(n_estimators, max_depth):
    result = []
    for estimator in n_estimators:
        for depth in max_depth:
            wmaes_cv = []
            for i in range(1,5):
                print('k:', i, ', n_estimators:', estimator, ', max_depth:', depth)
                x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.3)
                RF = RandomForestRegressor(n_estimators=estimator, max_depth=depth)
                RF.fit(x_train, y_train)
                predicted = RF.predict(x_test)
                wmaes_cv.append(WMAE(x_test, y_test, predicted))
            print('WMAE:', np.mean(wmaes_cv))
            result.append({'Max_Depth': depth, 'Estimators': estimator, 'WMAE': np.mean(wmaes_cv)})
    return pd.DataFrame(result)
```
#### Result:
![wmae](https://github.com/Pradnya1208/Walmart-store-sales-prediction/blob/main/output/wmae.PNG?raw=true)

#### Tuning `Max features`, `min_samples_split` and `min_samples_leaf`:
```
RF = RandomForestRegressor(n_estimators=58, max_depth=27, max_features=6, min_samples_split=3, min_samples_leaf=1)
RF.fit(X_train, Y_train)
```
![maxfeatures](https://github.com/Pradnya1208/Walmart-store-sales-prediction/blob/main/output/maxfeatures.PNG?raw=true)

![splits](https://github.com/Pradnya1208/Walmart-store-sales-prediction/blob/main/output/splits.PNG?raw=true)


#### Final Model: 
```
RF = RandomForestRegressor(n_estimators=58, max_depth=27, max_features=6, min_samples_split=3, min_samples_leaf=1)
RF.fit(X_train, Y_train)
```





### Lessons Learned

`Sales Prediction`
`IRandom Forest`
`Parameter Tuning`



### Feedback

If you have any feedback, please reach out at pradnyapatil671@gmail.com


### 🚀 About Me
#### Hi, I'm Pradnya! 👋
I am an AI Enthusiast and  Data science & ML practitioner


[1]: https://github.com/Pradnya1208
[2]: https://www.linkedin.com/in/pradnya-patil-b049161ba/
[3]: https://public.tableau.com/app/profile/pradnya.patil3254#!/
[4]: https://twitter.com/Pradnya1208


[![github](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c292abd3f9cc647a7edc0061193f1523e9c05e1f/icons/git.svg)][1]
[![linkedin](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/9f5c4a255972275ced549ea6e34ef35019166944/icons/iconmonstr-linkedin-5.svg)][2]
[![tableau](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/e257c5d6cf02f13072429935b0828525c601414f/icons/icons8-tableau-software%20(1).svg)][3]
[![twitter](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c9f9c5dc4e24eff0143b3056708d24650cbccdde/icons/iconmonstr-twitter-5.svg)][4]



