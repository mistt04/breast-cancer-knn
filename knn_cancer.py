import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


df = pd.read_csv("Breast Cancer Wisconsin Data.csv")
df = df.drop(["id", "Unnamed: 32"], axis=1)
print(df.head())
df.info()
print(df.shape)

def diagnosis_value(diagnosis):
    if diagnosis == "M":
        return 1
    elif diagnosis == "B":
        return 0 

df["diagnosis"] = df["diagnosis"].apply(diagnosis_value)

sns.lmplot(x = "radius_mean", y = "texture_mean", hue = "diagnosis", data = df)
plt.show()
sns.lmplot(x = "smoothness_mean", y = "compactness_mean", hue = "diagnosis", data = df)
plt.show()

X = np.array(df.iloc[:, 1:])
y = np.array(df["diagnosis"])

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

knn = KNeighborsClassifier(n_neighbors = 13)
knn.fit(X_train, y_train)
knn.score(X_test, y_test)

neighbors = []
cv_scores = []

from sklearn.model_selection import cross_val_score
for k in range(1, 51, 2):
    neighbors.append(k)
    knn = KNeighborsClassifier(n_neighbors = k)
    scores = cross_val_score(
        knn, X_train, y_train, cv = 10, scoring = "accuracy"
    )
    cv_scores.append(scores.mean())

MSE = [1 - x for x in cv_scores]
optimal_k = neighbors[MSE.index(min(MSE))]
print(f"The optimal number of neighbors is {optimal_k}")

plt.figure(figsize = (10, 6))
plt.plot(neighbors, MSE)
plt.title("Miscalculation Error vs Number of Neighbors as K")
plt.xlabel("Number of neighbors")
plt.ylabel("Miscalculation Error")
plt.show()