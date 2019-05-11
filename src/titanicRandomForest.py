
# Data analysis
import pandas as pd
# Prep data
from sklearn.impute import SimpleImputer
# Model
from sklearn.ensemble import RandomForestClassifier

# Let's read in the data first
trainFile = '../input/train.csv'
testFile = '../input/test.csv'
trainData = pd.read_csv(trainFile)
testData = pd.read_csv(testFile)

# First we want to get rid of the columns with too many missing entries
trainData = trainData.drop(['Ticket', 'Cabin'], axis=1)
testData = testData.drop(['Ticket', 'Cabin'], axis=1)
# Fill in the single missing fare with the median
testData['Fare'].fillna(testData['Fare'].dropna().median(), inplace=True)
combine = [trainData, testData]
# Embarked feature takes S, Q, C values based on port of embarkation. 
# Our training dataset has two missing values. 
# We simply fill these with the most common occurrence.
freq_port = trainData.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
# We have a name column, but a person's name is not useful in itself, we will extract only the person's title
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    
# We will replace the rare titles with a single title 'Rare'
# We will replace some of the other titles with a more common name
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
     'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
# Title, Sex, and Embarked are categorical values
# We will replace them with one hot encodings
cols_to_transform = [ 'Title', 'Sex', 'Embarked' ]
trainData = pd.get_dummies(trainData, columns = cols_to_transform)
testData = pd.get_dummies(testData, columns = cols_to_transform)
    
# Drop the Name and PassengerId columns as they are not useful in classification
trainData = trainData.drop(['Name', 'PassengerId'], axis=1)
testData = testData.drop(['Name'], axis=1)
# We will be filling in any remaining missing values using imputation
myImputer = SimpleImputer()
imputedTrain = myImputer.fit_transform(trainData)
imputedTest = myImputer.transform(testData)
# Imputation creates np arrays, we want the data as data frames
trainData = pd.DataFrame(imputedTrain, columns = trainData.columns)
testData = pd.DataFrame(imputedTest, columns = testData.columns)
combine = [trainData, testData]

# Convert the age and fare features based on ranges chosen during exploration
# Convert Age to ordinal values
# pd.cut cuts the total age range into equal bins based on the age range
# so each bin would cover age range of size (maxAge - minAge) / numbins
for dataset in combine:    
    dataset['Age'] = pd.cut(dataset.Age, 5, labels=False)
# Convert Fare to ordinal values
# pd.qcut cuts the fare range into equals bins based on the frequency of values,
# so each bin contains a similar number of values, to deal with outliers
for dataset in combine:
    dataset['Fare'] = pd.qcut(dataset.Fare, 6, labels=False)

# We have two features showing number of parents and siblings on the ship
# We will combine them to create a feature showing whether a person was
# traveling alone
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[(dataset['SibSp'] + dataset['Parch']) == 0, 'IsAlone'] = 1
# Remove the SibSp and Parch features
trainData = trainData.drop(['Parch', 'SibSp'], axis=1)
testData = testData.drop(['Parch', 'SibSp'], axis=1)
combine = [trainData, testData]

# Now the model
X_train = trainData.drop("Survived", axis=1)
Y_train = trainData["Survived"]
X_test  = testData.drop("PassengerId", axis=1).copy()
# Random Forest Classifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
# Data frame contains floats, but the output needs to be of type int
submission = pd.DataFrame({
        "PassengerId": testData["PassengerId"].astype(int),
        "Survived": Y_pred.astype(int)
    })
submission.to_csv('../output/submission.csv', index=False)








