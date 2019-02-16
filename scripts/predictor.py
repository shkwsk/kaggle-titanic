import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# load training data
train_df = pd.read_csv('../input/train.csv', header=0)

# convert sex to categorical number
train_df['Gender'] = train_df['Sex'].map({'female':0, 'male':1}).astype(int)
train_df.head(3)
