# Project Name
> Outline a brief description of your project.


## Table of Contents
* [Prerequisite & Objectives](#general-information)
* [Analysis Steps](#Data Analysis)
* [Multiple Linear Regression Methods](#Model Building)
* [Python Libraries](#technologies-used)
* [Conclusions & Recommendations](#conclusions)
* [Acknowledgements](#acknowledgements)
<!-- You can include any other section that is pertinent to your problem -->

## General Information
- This assignment is a programming assignment wherein we have to build a multiple linear regression model for the prediction of demand for shared bikes.
- A bike-sharing system is a service in which bikes are made available for shared use to individuals on a short term basis for a price or free. Many bike share systems allow people to borrow a bike from a "dock" which is usually computer-controlled wherein the user enters the payment information, and the system unlocks it. This bike can then be returned to another dock belonging to the same system.


A US bike-sharing provider BoomBikes has recently suffered considerable dips in their revenues due to the ongoing Corona pandemic. The company is finding it very difficult to sustain in the current market scenario. So, it has decided to come up with a mindful business plan to be able to accelerate its revenue as soon as the ongoing lockdown comes to an end, and the economy restores to a healthy state. 


In such an attempt, BoomBikes aspires to understand the demand for shared bikes among the people after this ongoing quarantine situation ends across the nation due to Covid-19. They have planned this to prepare themselves to cater to the people's needs once the situation gets better all around and stand out from other service providers and make huge profits.


They have contracted a consulting company to understand the factors on which the demand for these shared bikes depends. Specifically, they want to understand the factors affecting the demand for these shared bikes in the American market. The company wants to know:

Which variables are significant in predicting the demand for shared bikes.
How well those variables describe the bike demands
Based on various meteorological surveys and people's styles, the service provider firm has gathered a large dataset on daily bike demands across the American market based on some factors. 


- We are required to model the demand for shared bikes with the available independent variables. It will be used by the management to understand how exactly the demands vary with different features. They can accordingly manipulate the business strategy to meet the demand levels and meet the customer's expectations. Further, the model will be a good way for management to understand the demand dynamics of a new market. 
- We can observe in the dataset that some of the variables like 'weathersit' and 'season' have values as 1, 2, 3, 4 which have specific labels associated with them (as can be seen in the data dictionary). These numeric values associated with the labels may indicate that there is some order to them - which is actually not the case (Check the data dictionary and think why). So, it is advisable to convert such feature values into categorical string values before proceeding with model building. Please refer the data dictionary to get a better understanding of all the independent variables.
 
We notice the column 'yr' with two values 0 and 1 indicating the years 2018 and 2019 respectively. At the first instinct, you might think it is a good idea to drop this column as it only has two values so it might not be a value-add to the model. But in reality, since these bike-sharing systems are slowly gaining popularity, the demand for these bikes is increasing every year proving that the column 'yr' might be a good variable for prediction. So we took decision before dropping it. 
 

Model Building

In the dataset provided, you will notice that there are three columns named 'casual', 'registered', and 'cnt'. The variable 'casual' indicates the number casual users who have made a rental. The variable 'registered' on the other hand shows the total number of registered users who have made a booking on a given day. Finally, the 'cnt' variable indicates the total number of bike rentals, including both casual and registered. The model should be built taking this 'cnt' as the target variable.


Model Evaluation:
When we're done with model building and residual analysis and have made predictions on the test set, we makemade sure to use the following two lines of code to calculate the R-squared score on the test set.

 

from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
 

where y_test is the test data set for the target variable, and y_pred is the variable containing the predicted values of the target variable on the test set.

<!-- You don't have to answer all the questions - just the ones relevant to your project. -->

## Conclusions
- Analysing the above model, the comapany should focus on the following features:
- Company should focus on expanding business during Spring.
- Company should focus on expanding business during September.
- Based on previous data it is expected to have a boom in number of users once situation comes back to normal, compared to 2019.
- There would be less bookings during Light Snow or Rain, they could probably use this time to serive the bikes without having business impact.
- Hence when the situation comes back to normal, the company should come up with new offers during spring when the weather is pleasant and also advertise a little for September as this is when business would be at its best.
- Significant variables to predict the demand for shared bikes

holiday
temp
hum
windspeed
Season
months(January, July, September, November, December)
Year (2019)
Sunday
weathersit( Light Snow, Mist + Cloudy)

<!-- You don't have to answer all the questions - just the ones relevant to your project. -->


## Technologies Used
- Python library - numpy
- Python library - pandas
- Python library - matplotlib.pyplot
- Python library - seaborn
- Python library - datetime
- from sklearn.model_selection import train_test_split
- from sklearn.preprocessing import StandardScaler
- import statsmodels.api as sm
- from statsmodels.stats.outliers_influence import variance_inflation_factor
- from sklearn.linear_model import LinearRegression
- from sklearn.feature_selection import RFE
- from sklearn.model_selection import train_test_split
- from sklearn.metrics import r2_score
- from sklearn.preprocessing import MinMaxScaler

<!-- As the libraries versions keep on changing, it is recommended to mention the version of library used in this project -->

## Acknowledgements
Give credit here.
- This project was inspired by MS program in AI/ML from IIITB and LJMU which is conducted by Upgrad as an assignment
- Wikipedia in general
- Guidance and outline provided by  Mirza Rahim Baig Lead Business Analyst at Flipkart:


## Contact
Created by [@AnjanKumarPal1] - feel free to contact me!


<!-- Optional -->
<!-- ## License -->
<!-- This project is open source and available under the [... License](). -->

<!-- You don't have to include all sections - just the one's relevant to your project -->