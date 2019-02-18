import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# load data
train_df = pd.read_csv('../input/train.csv', header=0)
test_df = pd.read_csv('../input/test.csv', header=0)

# convert sex to categorical number
train_df['Gender'] = train_df['Sex'].map({'female':0, 'male':1}).astype(int)
test_df['Gender'] = test_df['Sex'].map({'female':0, 'male':1}).astype(int)
print(train_df.head(3))
print(test_df.head(3))

# complement the missing values of age column with average of age
median_age = train_df['Age'].dropna().median()
if len(train_df.Age[train_df.Age.isnull()]) > 0:
    train_df.loc[(train_df.Age.isnull()), 'Age'] = median_age

median_age = test_df['Age'].dropna().median()
if len(test_df.Age[test_df.Age.isnull()]) > 0:
    test_df.loc[(test_df.Age.isnull()), 'Age'] = median_age

# copy id
ids = test_df.PassengerId.values

# remove un-used columns
train_df = train_df.drop(['Name','Ticket','Sex','SibSp','Parch','Fare','Cabin','Embarked','PassengerId'], axis=1)
test_df = test_df.drop(['Name','Ticket','Sex','SibSp','Parch','Fare','Cabin','Embarked','PassengerId'], axis=1)
print(train_df.head(3))
print(test_df.head(3))

# predict with random forest
train_data = train_df.values
test_data = test_df.values
model = RandomForestClassifier(n_estimators=100)
survived = model.fit(train_data[0::, 1::], train_data[0::, 0]).predict(test_data).astype(int)

# export result
output_df = pd.DataFrame({
    'PassengerId': ids,
    'Survived': survived
})
output_df.to_csv('../output/titanic_submit.csv', index=False)
