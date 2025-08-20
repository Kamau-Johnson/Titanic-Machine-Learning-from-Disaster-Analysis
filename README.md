
# Titanic Machine Learning from Disaster.
![Screenshot 18](Screenshot%2018.png)
### Introduction.

The Titanic disaster of 1912 claimed the lives of more than 1,500 passengers and crew, making it one of the deadliest maritime tragedies in history. Historical accounts suggest that survival chances were not random. Factors such as gender, age, and social class played a major role, with the “women and children first” policy influencing who made it into the lifeboats.
This project uses the Titanic dataset from Kaggle to explore these patterns and build a machine learning model that can predict whether a passenger survived based on their personal and travel details. By analyzing demographic information, ticket class, fares paid, and family relationships aboard the ship, the goal is to uncover which factors most strongly influenced survival and to create a model that can make accurate predictions. The results provide insight into the historical event and show how data science can be applied to understand real-world outcomes.

### Problem Statement.
The objective of this project is to use the Titanic passenger dataset to develop a predictive model that can determine the likelihood of survival for each passenger based on available information. The aim is to combine historical understanding with data-driven analysis by exploring patterns in the data, identifying the most influential factors affecting survival, and applying machine learning techniques to make accurate predictions.
Through this process, the project seeks to demonstrate the complete data science workflow, including data exploration, cleaning, feature engineering, model training, and evaluation. The final goal is not only to create a reliable survival prediction model but also to gain meaningful insights into the factors that shaped the survival outcomes of Titanic passengers.

### Scope of the Analysis.
This project focuses on analyzing the Titanic passenger dataset to build and evaluate a predictive model for survival. It aims to explore patterns in the data, clean and prepare the dataset, create new features, and apply machine learning models to identify the most important factors affecting survival. The scope is limited to the variables provided in the dataset and does not include any external data sources.

### Dataset Description.
#### Overview of the Data.
The dataset used in this analysis is the Titanic passenger dataset from Kaggle. It contains historical records of passengers aboard the RMS Titanic, including demographic details, ticket and cabin information, travel class, and survival status. The training dataset has 891 passenger records with both feature values and survival labels, while the test dataset has 418 records with feature values only, intended for prediction.

### Data Characteristics.
The dataset includes both numerical and categorical variables. Numerical variables include Age, Fare, SibSp (number of siblings/spouses aboard), and Parch (number of parents/children aboard). Categorical variables include Sex, Pclass (passenger class), and Embarked (port of embarkation). Some features contain missing values, particularly Age, Cabin, and Embarked, requiring data cleaning and imputation. Additional engineered features such as FamilySize, IsAlone, and Has Embarked were created to enhance model performance.

### Data Preprocessing.
Before modeling, the dataset was prepared through several preprocessing steps to ensure data quality and consistency.
#### Structure of the Workflow:
I started my data analysis by loading my csv files into my model using the pandas library loading the gender_submisson.csv, test.csv and the tran.csv. This allowed me to have the training data, the test data for predictions, and a sample submission file for data analysis :

![Screenshot 1](Screenshot%201.png)

After loading the datasets, I performed an initial inspection using functions such as .head() to view the first few rows, .info() to check data types and non-null counts, and .describe() to see basic statistical summaries of the numerical features :

![Screenshot 2](Screenshot%202.png)
![Screenshot 3](Screenshot%203.png)
![Screenshot 4](Screenshot%204.png)

The next step was to explore the dataset for missing values using .isnull().sum(). This revealed that the Age, Cabin, and Embarked columns had missing entries. Based on this finding, I applied specific cleaning strategies: filling missing Embarked values with the most common category, imputing Age with the median based on passenger sex and class, filling missing Fare values in the test set with the median, and dropping the Cabin column due to excessive missing data :

![Screenshot 5](Screenshot%205.png)

Once the data was cleaned, I proceeded with feature engineering to enhance the dataset. I created FamilySize by adding SibSp and Parch and including the passenger, IsAlone to indicate passengers traveling without family, and Has Embarked to flag whether cabin data was present.

![Screenshot 6](Screenshot%206.png)

In this I started approaches of statistical ploting of graphs for easier visualization of data :

![Screenshot 7](Screenshot%207.png)

The graph below illustrates the relationship between sex and survival in the dataset. It shows that survival rates vary notably between males and females, suggesting that gender may have played an important role in determining the likelihood of survival. This insight helps us understand demographic patterns in the data.

![Screenshot 8](Screenshot%208.png)

The graph illustrates the relationship between passenger class (Pclass) and survival. It reveals that survival rates were higher in the upper classes and declined as class level decreased, indicating a possible link between socio-economic status and chances of survival: 

![Screenshot 9](Screenshot%209.png)

In addition to analyzing sex and passenger class, I conducted further exploratory analyses on other variables and built predictive models to better understand the factors affecting survival. These additional steps helped validate the insights and improve the overall accuracy of the findings.

## Exploratory Data Analysis (EDA).
In this section, I performed a detailed exploratory analysis to uncover key patterns and relationships in the data. Some of the main insights include:
- Sex and Survival: Females had a notably higher survival rate compared to males, indicating gender was an important factor.
- Passenger Class (Pclass) and Survival: Passengers in higher classes had better survival chances, suggesting socio-economic status impacted outcomes.
- Other Variables: Additional analysis was conducted on age, fare, and embarked location, among others, to better understand their influence on survival.
These insights helped guide the development of predictive models and deeper analysis.


## Feature Engineering.
- Family Size: Created by combining the number of siblings/spouses (SibSp) and parents/children (Parch) aboard to capture the effect of traveling with family on survival chances. Larger families might have different survival patterns compared to individuals.
- Is Alone: A binary feature derived from Family Size, indicating whether a passenger was traveling alone. This helps assess if traveling solo influenced survival outcomes.
- Title Extraction: Extracted titles (Mr., Mrs., Miss, etc.) from passenger names to capture social status or age group differences that might affect survival.
- Age Groups: Categorized continuous age values into groups (e.g., child, adult, senior) to better handle non-linear relationships with survival.
Each new feature was created to enhance model performance by incorporating meaningful information not directly available from the original variables.

- Categorical variables such as Sex and Embarked were encoded into numerical values to be compatible with machine learning algorithms. I then split the cleaned dataset into training and validation sets to allow for model evaluation by use of graphs and other various techniques :

![Screenshot 10](Screenshot%2010.png)

![Screenshot 11](Screenshot%2011.png)

## Model Training.
I first split the dataset into training and validation sets to properly train and evaluate the models. After training the initial model, the validation accuracy obtained was 0.8156424581005587 (approximately 81.56%). Building on this, I proceeded to apply more advanced models for better prediction.

![Screenshot 12](Screenshot%2012.png)

For modeling, I started with Logistic Regression because it is easy to understand and works well for predicting two possible outcomes. After that, I used a Random Forest Classifier, which is a more advanced model that combines many decision trees to make better predictions. I checked how well both models performed by looking at their accuracy on the validation data, and both models scored about 81.56% accuracy.

![Screenshot 13](Screenshot%2013.png)

- The Model gave us a validation accuracy of  0.8156424581005587 which was actually accurate.

Finally, I looked at the Random Forest model to see which features were most important in predicting survival. The results showed that Sex, Age, and Fare were the top factors that influenced whether a passenger survived or not.

![Screenshot 14](Screenshot%2014.png)

- I confirmed the previous Logistics regression if my model was actually correct in the training, prediction and validation stage and the model predicted accurately with the same validation accuracy of  0.8156424581005587.

Later, I performed a feature importance analysis using the Random Forest model to better understand which variables had the strongest impact on predicting survival. Random Forest works by building many decision trees and combining their results, which makes it very effective at capturing complex relationships in the data. One of its advantages is that it provides a measure of how much each feature contributes to the model’s decisions.

![Screenshot 15](Screenshot%2015.png)

![Screenshot 16](Screenshot%2016.png)

![Screenshot 17](Screenshot%2017.png)
## Documentation

[Documentation](https://medium.com/@Kamau_Johnson/titanic-machine-learning-from-disaster-5cbeb699cbb6)



## Features

- Applies key course concepts to a real-world problem

- Organized and well-documented code for easy understanding

- Clear, step-by-step problem-solving approach from data to model


## Running Tests

To evaluate the machine learning models and validate their performance, you can run the test scripts included in the project. For example, if using pytest, run:

```bash
pytest test_model.py

```
This will execute all the tests related to model accuracy, data preprocessing, and prediction consistency.

If you created custom test scripts, ensure they cover:

- Data preprocessing correctness

- Model training and validation accuracy

- Prediction outputs on sample inputs


## Tech Stack

**Programming Language:** Python

**Data Handling and Analysis:** Pandas & Numpy

**Data Visualization:** Matplotlib & Seaborn 

**Machine Learning::** Scikit-learn

**Environment and Tools:** Jupyter Notebook * Git 








## Authors

- Kamau Johnson

## Lessons Learned

Working on the Titanic survival prediction project deepened my understanding of the complete data science workflow from data cleaning and feature engineering to model building and evaluation. I learned how crucial thorough data preprocessing is, especially handling missing values and creating meaningful features like Family Size and Title extraction to improve model performance.

A key challenge was balancing model simplicity with accuracy. Starting with Logistic Regression helped build intuition, while exploring Random Forests revealed the power of ensemble methods in capturing complex patterns. Evaluating model performance and interpreting feature importance reinforced the value of combining statistical insight with machine learning.

Overall, this project enhanced my skills in applying machine learning algorithms to real-world data, interpreting model results, and communicating findings clearly, preparing me for more advanced data science challenges.

## Feedback

For feedback or suggestions, feel free to reach out via my portfolio at https://kamaujohnson.dev. I’m happy to connect on topics related to data science or software development.
## Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://kamaujohnson.dev/)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kamau-johnson-4bab25276/)
[![medium](https://img.shields.io/badge/Medium-000000?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/@Kamau_Johnson)
[![YouTube](https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@Kamau_Johnson)

### Byeee !!!
