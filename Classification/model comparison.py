import numpy as np
import pandas as pd
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier

# drop country information if the country code does not appear in both files
def drop_data(df, code):
    for index, row in df.iterrows():
        if row['Country Code'] not in code:
            df = df.drop(index, axis=0)
    return df

world = pd.read_csv('world.csv', encoding = "ISO-8859-1")
life = pd.read_csv('life.csv', encoding = "ISO-8859-1")

# extract common country codes appear in both files
common_code = list(set(world['Country Code']) & set(life['Country Code']))

# pre-processing section
world = drop_data(world, common_code)
life = drop_data(life, common_code)

world = world.sort_values(by=['Country Code'])
life = life.sort_values(by=['Country Code'])

world.reset_index(drop=True, inplace=True)
life.reset_index(drop=True, inplace=True)

world = world.replace('..', np.NaN)

# discard country names, etc.
# start classification
data = world.iloc[:, 3:].astype(float)

classlabel = life['Life expectancy at birth (years)']

x_train, x_test, y_train, y_test = train_test_split(data, classlabel, train_size = 0.7, test_size = 0.3, random_state = 200)

# prepare columns for csv output
feature = world.columns[3:]
median = x_train.median()
x_train = x_train.fillna(median)
x_test = x_test.fillna(median)

# standarlisation
scaler = preprocessing.StandardScaler().fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

mean = scaler.mean_
variance = scaler.var_

# decision tree
dt = DecisionTreeClassifier(random_state = 200, max_depth = 3)
dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)
print("Accuracy of decision tree:", round(accuracy_score(y_test, y_pred), 3))

# 3-NN
knn = neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
y_pred=knn.predict(x_test)
print("Accuracy of k-nn (k=3):", round(accuracy_score(y_test, y_pred), 3))

# 7-NN
knn  = neighbors.KNeighborsClassifier(n_neighbors=7)
knn.fit(x_train, y_train)
y_pred=knn.predict(x_test)
print("Accuracy of k-nn (k=7):", round(accuracy_score(y_test, y_pred), 3))

# formating and output csv files
median = ['%.3f' % num for num in median]
mean = ['%.3f' % num for num in mean]
variance = ['%.3f' % num for num in variance]

task2a = pd.DataFrame({'feature': feature, 'median': median, 'mean': mean, 'variance': variance})
task2a.to_csv('task2a.csv', index = False)




