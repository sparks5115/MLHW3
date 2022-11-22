import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from itertools import combinations


weekly = pd.read_csv("Datasets\Weekly.csv")

# a
weekly = weekly.dropna()
weekly["Direction"] = weekly["Direction"].map({"Up": 1, "Down": 0})
sns.set_style("whitegrid")
sns.pairplot(weekly, height=1)
plt.show()


# b
# preform logistic regression on Direction using all other variables as predictors

X = weekly.drop(["Direction", "Year", "Today"], axis=1)
y = weekly["Direction"]
X = sm.add_constant(X)
model = sm.Logit(y, X).fit()
# find the p values for each variable
print(model.summary())
print()

# c
# find the confusion matrix
print("\nConfusion Matrix")
y_pred = model.predict(X)
y_pred = y_pred.map(lambda x: 1 if x > 0.5 else 0)
print(pd.crosstab(y, y_pred))
print()

# d
weekly2008 = weekly[weekly["Year"] < 2009]
X = weekly2008["Lag2"]
X = sm.add_constant(X)
y = weekly2008["Direction"]
model = sm.Logit(y, X).fit()
print("\nConfusion Matrix for <= 2008 wtih Lag2")
y_pred = model.predict(X)
y_pred = y_pred.map(lambda x: 1 if x > 0.5 else 0)
print(pd.crosstab(y, y_pred))
print("Logistic Regression accuracy: ", (y == y_pred).sum() / len(y))
print()


# e
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

model = LinearDiscriminantAnalysis().fit(X, y)
print("\nConfusion Matrix for <= 2008 wtih Lag2 using LDA")
y_pred = model.predict(X)
print(pd.crosstab(y, y_pred))
print("LDA accuracy: ", model.score(X, y))
print()

# f
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

model = QuadraticDiscriminantAnalysis().fit(X, y)
print("\nConfusion Matrix for <= 2008 wtih Lag2 using QDA")
y_pred = model.predict(X)
print(pd.crosstab(y, y_pred))
print("QDA accuracy: ", model.score(X, y))
print()

# g
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=1).fit(X, y)
print("\nConfusion Matrix for <= 2008 wtih Lag2 using KNN K=1")
y_pred = model.predict(X)
print(pd.crosstab(y, y_pred))
print("KNN K=1 accuracy: ", model.score(X, y))
print()

# h
# find an ideal K value when using 10 fold cross validation
k_range = range(1, 150)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring="accuracy")
    k_scores.append(scores.mean())
plt.plot(k_range, k_scores)
plt.xlabel("Value of K for KNN")
plt.ylabel("Cross-Validated Accuracy")
plt.show()

# use knn with k=34 (found to be the best value)
knn = KNeighborsClassifier(n_neighbors=34).fit(X, y)
scores = cross_val_score(knn, X, y, cv=10, scoring="accuracy")
print("KNN K=34 accuracy: ", scores.mean())  # KNN K=34 accuracy:  0.5634817563388992
print()

# try knn k=34 all combinations of predictors
possible_predictors = weekly2008.drop(["Direction", "Year", "Today"], axis=1).columns
for i in range(1, len(possible_predictors)):
    for predictors in combinations(possible_predictors, i):
        X = weekly2008[list(predictors)]
        X = sm.add_constant(X)
        y = weekly2008["Direction"]
        knn = KNeighborsClassifier(n_neighbors=34).fit(X, y)
        print("Predictors:", predictors)
        scores = cross_val_score(knn, X, y, cv=10, scoring="accuracy")
        print("\tKNN K=34 accuracy: ", scores.mean())
        print()

# use logistic regression with all predictors
possible_predictors = weekly2008.drop(["Direction", "Year", "Today"], axis=1).columns
best_predictors = []
best_score = 0
for i in range(1, len(possible_predictors)):
    for predictors in combinations(possible_predictors, i):
        X = weekly2008[list(predictors)]
        X = sm.add_constant(X)
        model = LogisticRegression().fit(X, y)
        scores = cross_val_score(model, X, y, cv=10, scoring="accuracy")
        if scores.mean() > best_score:
            best_predictors = predictors
            best_score = scores.mean()
print("Best predictors:", best_predictors)
print("\tLogistic Regression accuracy: ", best_score)
print()

# use logistic regression with log of lag2
temp = weekly2008.copy()
temp["Lag2"] = np.log(weekly2008["Lag2"])
# clean up the data
temp = temp[temp["Lag2"] != -np.inf]
temp = temp[temp["Lag2"] != np.inf]
temp = temp.dropna()
X = temp["Lag2"]
X = sm.add_constant(X)
y = temp["Direction"]
model = LogisticRegression().fit(X, y)
scores = cross_val_score(model, X, y, cv=10, scoring="accuracy")
print("Logistic Regression with log of Lag2 accuracy: ", scores.mean())
print()

# use logistic regression with log of lag2 and lag5
X = temp[["Lag2", "Lag5"]]
X = sm.add_constant(X)
y = temp["Direction"]
model = LogisticRegression().fit(X, y)
scores = cross_val_score(model, X, y, cv=10, scoring="accuracy")
print("Logistic Regression with log of Lag2 and Lag5 accuracy: ", scores.mean())
print()

# use logistic regression with lag2^2
temp = weekly2008.copy()
temp["Lag2"] = weekly2008["Lag2"] ** 2
# clean up the data
temp = temp[temp["Lag2"] != -np.inf]
temp = temp[temp["Lag2"] != np.inf]
temp = temp.dropna()
X = temp["Lag2"]
X = sm.add_constant(X)
y = temp["Direction"]
model = LogisticRegression().fit(X, y)
scores = cross_val_score(model, X, y, cv=10, scoring="accuracy")
print("Logistic Regression with Lag2^2 accuracy: ", scores.mean())
print()

# use logistic regression with lag2^2 and lag5^2
X = temp[["Lag2", "Lag5"]]
X = sm.add_constant(X)
y = temp["Direction"]
model = LogisticRegression().fit(X, y)
scores = cross_val_score(model, X, y, cv=10, scoring="accuracy")
print("Logistic Regression with Lag2^2 and Lag5^2 accuracy: ", scores.mean())
print()
