import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import metrics


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

# imputation for missing value
median = data.median()
data = data.fillna(median)

# start method 1, interaction pairs and clustering
data.columns = list(range(20))

# plot graph1 that justify my choice of cluster number
distortion = []
for k in range(1, 10):
    kmean = KMeans(n_clusters = k)
    kmean.fit(data)
    distortion.append(kmean.inertia_)

plt.figure(0)
plt.plot(range(1, 10), distortion, 'bx-')
plt.xlabel('k number')
plt.ylabel('distortion')
plt.title('Elbow Method finding optimal k')
plt.savefig('task2bgraph1.png')

print("Visualising the graph, I determine the best cluster number is 3")

# generate clusters
# take 3 as cluster number
kmean = KMeans(n_clusters = 3)
clusters = kmean.fit(data)
data[210] = clusters.labels_

# generate interaction column
col_num = 20
for i in range(20):
    for j in range(i+1, 20):
        col_1 = data[i]
        col_2 = data[j]
        data[col_num] = col_1 * col_2
        col_num += 1;

# use mutual information to compute scores for each feature
scores = []
for i in range(211):    
    disc = pd.qcut(data[i], 15, labels=False, duplicates = 'drop')
    score = metrics.mutual_info_score(disc, classlabel)
    scores.append(score)  

# select 4 features with top 4 scores
maxpos = []
maxscores = []
for i in range(4):
    maxpo = scores.index(max(scores))
    maxscores.append(max(scores))
    maxpos.append(maxpo)
    del scores[maxpo]

print('The four features being selected is:')
print(maxpos)

print('Their corresponding mutual information score is:')
print(maxscores)
# start train and test   
avg_1 = []


knn = neighbors.KNeighborsClassifier(n_neighbors=3)
selected = data[maxpos]

# compute average accuracy for method
for i in range(100):
    x_train, x_test, y_train, y_test = train_test_split(selected, classlabel, train_size = 0.7, test_size = 0.3)

    # standarlisation
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train=scaler.transform(x_train)
    x_test=scaler.transform(x_test)

    knn.fit(x_train, y_train)

    y_pred=knn.predict(x_test)
    avg_1.append(accuracy_score(y_test, y_pred))

avg_1 = pd.Series(avg_1)
print("Accuracy of feature engineering:", round(avg_1.mean(), 3))
    
# start method 2, PCA
# standarlise data 
scaler = preprocessing.StandardScaler().fit(data)
data_std = scaler.transform(data)

pca = PCA(n_components = 4)
principalComponents = pca.fit_transform(data_std)
principalDf = pd.DataFrame(data = principalComponents
        , columns = ['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4'])

# train and test
# compute average accuracy for this method
avg_2 = []
knn = neighbors.KNeighborsClassifier(n_neighbors=3)
for i in range(100):
    x_train, x_test, y_train, y_test = train_test_split(principalDf, classlabel, train_size = 0.7, test_size = 0.3)

    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train=scaler.transform(x_train)
    x_test=scaler.transform(x_test)
    
    knn.fit(x_train, y_train)

    y_pred=knn.predict(x_test)
    avg_2.append(accuracy_score(y_test, y_pred))

avg_2 = pd.Series(avg_2)
print("Accuracy of PCA:", round(avg_2.mean(), 3))


# start method 3, select first four features
first_four = data.iloc[:, :4]

# train and test
# compute average accuracy for this method
avg_3 = []
knn = neighbors.KNeighborsClassifier(n_neighbors=3)
for i in range(100):
    x_train, x_test, y_train, y_test = train_test_split(first_four, classlabel, train_size = 0.7, test_size = 0.3)
    
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train=scaler.transform(x_train)
    x_test=scaler.transform(x_test)

    knn.fit(x_train, y_train)

    y_pred=knn.predict(x_test)
    avg_3.append(accuracy_score(y_test, y_pred))

avg_3 = pd.Series(avg_3)
print("Accuracy of first four features:", round(avg_3.mean(), 3))


