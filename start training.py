import os
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

test = "test.csv"
train = "train.csv"

header = ["buying_price", "maintenance_cost", "number_of_doors", "number_of_seats", "luggage_boot_size", "safety_rating"]

#dataset = pandas.read_csv(test, names=header)
dataset = pandas.read_csv(train)

#print dataset
#dataset.plot(kind='box', subplots=True, layout=(2,4), sharex=False, sharey=False)
#plt.show()

#dataset.hist()
#plt.show()

array = dataset.values

A = array[:,0:6]
B = array[:,6]

size = 0.20
random_seed = 7

A_train, A_validation, B_train, B_validation = model_selection.train_test_split(A, B, test_size=size, random_state=random_seed)

accuracy = "accuracy"

models = []
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('SVM', SVC()))
models.append(('Naive', GaussianNB()))

results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=15, random_state=random_seed)
    cv_results = model_selection.cross_val_score(model, A_train, B_train, cv=kfold, scoring=accuracy)
    results.append(cv_results)
    names.append(name)
    news = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(news)


fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

print

cart = DecisionTreeClassifier()
cart.fit(A_train, B_train)


svm = SVC()
svm.fit(A_train, B_train)


#print A_validation

test_dataset = pandas.read_csv(test, names=header).values

output = cart.predict(test_dataset)

print output

df = pandas.DataFrame(output)
df.to_csv("prediction.csv", index=None, header=None)
raw_input()