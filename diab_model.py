#import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import pandas_profiling
plt.style.use("ggplot")
import seaborn as sns
#import plotly.express as px
# %matplotlib inline
sns.set_palette("bwr")

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
import pickle

df = pd.read_csv('Diabets.csv')

X = df.iloc[:, :16]
y = pd.DataFrame(df.iloc[:,-1])


## Binary (indicator variable coding)
one_values = ["Male", "Positive", "Yes"]
zero_values = ["Female", "Negative", "No"]
for column in X.columns:
    X[column] = X[column].replace(to_replace=[one_values], value=1)
    X[column] = X[column].replace(to_replace=[zero_values], value=0)
for column in y.columns:
    y[column] = y[column].replace(to_replace=[one_values], value=1)
    y[column] = y[column].replace(to_replace=[zero_values], value=0)

# Split data into train and test set
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7)

decisiontree = DecisionTreeClassifier(max_depth=4)
decisiontree.fit(x_train, y_train)
#decisiontree_pred = decisiontree.predict(x_test)

# Saving model to disk
pickle.dump(decisiontree, open('diab_model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('diab_model.pkl','rb'))
print(model.predict(x_test))
