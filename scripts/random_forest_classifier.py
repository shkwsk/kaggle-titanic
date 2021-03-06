import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# load data
train_df = pd.read_csv('../input/train.csv', header=0)
test_df = pd.read_csv('../input/test.csv', header=0)

# convert to categorical number
# sex
train_df['Sex'] = train_df['Sex'].map({'female':0, 'male':1}).astype(int)
test_df['Sex'] = test_df['Sex'].map({'female':0, 'male':1}).astype(int)
# embarked
freq_port = train_df.Embarked.dropna().mode()[0]
train_df['Embarked'] = train_df['Embarked'].fillna(freq_port).map({'S':0,'C':1,'Q':2}).astype(int)
test_df['Embarked'] = test_df['Embarked'].fillna(freq_port).map({'S':0,'C':1,'Q':2}).astype(int)
# print(train_df.head(3))
# print(test_df.head(3))

# complement the missing values of column
# age
train_df.Age.fillna(train_df.Age.dropna().median(), inplace=True)
test_df.Age.fillna(test_df.Age.dropna().median(), inplace=True)
# fare
train_df.Fare.fillna(train_df.Fare.dropna().median(), inplace=True)
test_df.Fare.fillna(test_df.Fare.dropna().median(), inplace=True)

# copy id
ids = test_df.PassengerId.values

# remove un-used columns
train_df = train_df.drop(['Name','Ticket','Cabin','PassengerId'], axis=1)
test_df = test_df.drop(['Name','Ticket','Cabin','PassengerId'], axis=1)
print(train_df.head(3))
print(test_df.head(3))

# instance of random forest
rfc = RandomForestClassifier(n_estimators=100)

# training
x_train = train_df.drop('Survived', axis=1)
y_train = train_df['Survived']
rfc.fit(x_train, y_train)

# predict
survived = rfc.predict(test_df).astype(int)

# export result
output_df = pd.DataFrame({
    'PassengerId': ids,
    'Survived': survived
})
output_df.to_csv('../output/titanic_submit.csv', index=False)
