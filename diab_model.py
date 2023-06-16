#import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import pandas_profiling
plt.style.use("ggplot")
import seaborn as sns
#import plotly.express as px
# %matplotlib inline
sns.set_palette("bwr")

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
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
    X[column] = X[column].replace(to_replace=one_values, value=1)
    X[column] = X[column].replace(to_replace=zero_values, value=0)
for column in y.columns:
    y[column] = y[column].replace(to_replace=one_values, value=1)
    y[column] = y[column].replace(to_replace=zero_values, value=0)

# Split data into train and test set
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7)

decisiontree = RandomForestClassifier(max_depth=10)
decisiontree.fit(x_train, y_train)
#decisiontree_pred = decisiontree.predict(x_test)

# Saving model to disk
pickle.dump(decisiontree, open('diab_model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('diab_model.pkl','rb'))
print(model.predict(x_test))
print(model.classes_)

print(classification_report(y_test, model.predict(x_test)))

prediction = model.predict(x_test)
pred_prob = model.predict_proba(x_test)

# Get the index of the predicted class for the first sample
predicted_class_index = model.classes_.tolist().index(prediction[0])

# Access the probability for the predicted class
predicted_class_probability = pred_prob[0][predicted_class_index]


output = round(prediction[0], 2)
pred_proba = round(predicted_class_probability, 2)
#"""
if (output==0.00):
    print('The predicted diabetes status is {}'.format(output)+', with prodicted probability {}'.format(pred_proba)+';'+ ' ' +'This means, based on the information supplied: No early diabetes symptoms. Do not forget to keep doing exercise and eating recommended diet.')
else:
    print('The predicted result indicated that it is class  {}'.format(output)+', with prodicted probability {}'.format(pred_proba)+';'+ ' ' +'perhaps, early diabetes symptoms. We would like to reiterate you that this is based on the information provided. Consult a practitionner, maintain physical exercise, and healthy diet.')
