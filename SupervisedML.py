import sklearn
import pandas as pd
import matplotlib.pyplot as plot
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve



def main():
    '''Preprocessing Adult Income dataset'''

    adult_dataset = pd.read_csv("adult.data.csv", header=None, delimiter= ",")
    adult_dataset = adult_dataset.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    adult_dataset.columns = [
        "Age", "WorkClass", "fnlwgt", "Education", "EducationNum",
        "MaritalStatus", "Occupation", "Relationship", "Race", "Gender",
        "CapitalGain", "CapitalLoss", "HoursPerWeek", "NativeCountry", "Income"
    ]
    adult_dataset["Income"] = adult_dataset["Income"].map({ "<=50K": 0, ">50K": 1 })
    del adult_dataset["Education"]
    X1 = adult_dataset.drop("Income", axis = 1)

    y1 = adult_dataset["Income"]

    from sklearn import preprocessing

    le = preprocessing.LabelEncoder()
    X1 = X1.apply(le.fit_transform)

    '''Preprocessing Phishing dataset'''

    phish_dataset = pd.read_csv("phish.data.csv", header=None, delimiter=",")
    phish_dataset = phish_dataset.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    phish_dataset.columns = [
        "SFH", "popUpWindow", "SSLfinal_State", "Request_URL", "URL_of_Anchor", "web_traffic",
        "URL_Length", "age_of_domain", "having_IP_Address", "Result"
    ]
    phish_dataset["Result"] = phish_dataset["Result"].map({-1 : 0, 1: 1})
    y2 = phish_dataset["Result"]
    X2 = phish_dataset.drop("Result", axis=1)

    '''Creating Standardized Versions of X and y for Adult Income dataset(X1,y1) \
       and Phishing dataset (X2,y2)'''
    from sklearn.preprocessing import StandardScaler
    sc1 = StandardScaler()
    sc2 = StandardScaler()
    sc2.fit(X2)
    sc1.fit(X1)
    X1_std = sc1.transform(X1)
    X2_std = sc2.transform(X2)

    '''Creating Normalized min/max Versions of X and y for Adult Income dataset(X1,y1) \
           and Phishing dataset (X2,y2)'''
    from sklearn.preprocessing import normalize
    from sklearn.preprocessing import minmax_scale
    X1_min_norm = normalize(minmax_scale(X1))
    X2_min_norm = normalize(minmax_scale(X2))

    '''Creating 80/20 train/test for Adult data set'''
    X1_train, X1_test, y1_train, y1_test = train_test_split(
        X1, y1, test_size=0.2)

    '''Creating 80/20 train/test for Phishing data set'''
    X2_train, X2_test, y2_train, y2_test = train_test_split(
        X2, y2, test_size=0.2)

    '''Creating Decision tree model on Adult income train split'''
    print("Adult Income Decision Tree:")
    decision_Tree_1(X1_train, X1_test, y1_train, y1_test)

    '''Creating Neural Network model on Adult income train split'''
    print("Adult Income Neural Network:")
    neural_net_1(X1_train, X1_test, y1_train, y1_test)

    '''Creating Boosted Decision Tree model on Adult income train split'''
    print("Adult Income Boosted Decision Tree:")
    boosted_DT_1(X1_train, X1_test, y1_train, y1_test)

    '''Creating K-Nearest Neighbors model on Adult income train split'''
    print("Adult Income KNN:")
    kNN_1(X1_train, X1_test, y1_train, y1_test)

    '''Creating SVM model on Adult income train split'''
    print("Adult Income SVM:")
    SVM_1(X1_train, X1_test, y1_train, y1_test)

    '''Creating Decision tree model on Phishing train split'''
    print("Phishing Decision Tree:")
    decision_Tree_2(X2_train, X2_test, y2_train, y2_test)

    '''Creating Neural Network model on Phishing train split'''
    print("Adult Income Neural Network:")
    neural_net_2(X2_train, X2_test, y2_train, y2_test)

    '''Creating Boosted Decision Tree model on Phishing train split'''
    print("Adult Income Boosted Decision Tree:")
    boosted_DT_2(X2_train, X2_test, y2_train, y2_test)

    '''Creating K-Nearest Neighbors model on Phishing train split'''
    print("Adult Income KNN:")
    kNN_2(X2_train, X2_test, y2_train, y2_test)

    '''Creating SVM model on Phishing train split'''
    print("Adult Income SVM:")
    SVM_2(X2_train, X2_test, y2_train, y2_test)

    '''Set up Learning curve step sizes'''
    training_sizes_accuracy =[]
    for i in range(50, 1000, 5):
        training_sizes_accuracy.append(i)
    training_sizes_f1 = []
    for i in range(600, 26000, 1000):
        training_sizes_f1.append(i)

    '''Learning curve for Adult decision Tree'''
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(max_depth=9)

    train_sizes_DT, train_scores_DT, validation_scores_DT = learning_curve(estimator=classifier,
                                                                      X=X1_std, y=y1,train_sizes=training_sizes_f1,
                                                                      scoring='f1', cv=5)
    train_scores_mean_DT = train_scores_DT.mean(axis=1)
    validation_scores_mean_DT = validation_scores_DT.mean(axis=1)
    plot_lc_f1(title='Adult Income: f1 Score/Training set size (DT)', train_sizes= training_sizes_f1,
               train_scores_mean= train_scores_mean_DT, validation_scores_mean=validation_scores_mean_DT)

    '''Learning curve for Adult Neural Network'''
    from sklearn.neural_network import MLPClassifier
    classifier = MLPClassifier((95,95))

    train_sizes_NN, train_scores_NN, validation_scores_NN = learning_curve(estimator=classifier,
                                                                           X=X1_min_norm, y=y1,
                                                                           train_sizes=training_sizes_f1,
                                                                           scoring='f1', cv=5)
    train_scores_mean_NN = train_scores_NN.mean(axis=1)
    validation_scores_mean_NN = validation_scores_NN.mean(axis=1)
    plot_lc_f1(title='Adult Income: f1 Score/Training set size (NN)', train_sizes=training_sizes_f1,
               train_scores_mean=train_scores_mean_NN, validation_scores_mean=validation_scores_mean_NN)

    '''Learning curve for Adult Boosted Decision Tree'''
    params = {'n_estimators': 50, 'max_depth': 9, 'min_samples_split': 2}
    from sklearn.ensemble import GradientBoostingClassifier
    classifier = GradientBoostingClassifier(**params)

    train_sizes_BDT, train_scores_BDT, validation_scores_BDT = learning_curve(estimator=classifier,
                                                                           X=X1_std, y=y1,
                                                                           train_sizes=training_sizes_f1,
                                                                           scoring='f1', cv=5)
    train_scores_mean_BDT = train_scores_BDT.mean(axis=1)
    validation_scores_mean_BDT = validation_scores_BDT.mean(axis=1)
    plot_lc_f1(title='Adult Income: f1 Score/Training set size (BDT)', train_sizes=training_sizes_f1,
               train_scores_mean=train_scores_mean_BDT, validation_scores_mean=validation_scores_mean_BDT)

    '''Learning curve for Adult KNN'''
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors=13)

    train_sizes_kNN, train_scores_kNN, validation_scores_kNN = learning_curve(estimator=classifier,
                                                                           X=X1_min_norm, y=y1,
                                                                           train_sizes=training_sizes_f1,
                                                                           scoring='f1', cv=5)
    train_scores_mean_kNN = train_scores_kNN.mean(axis=1)
    validation_scores_mean_kNN = validation_scores_kNN.mean(axis=1)
    plot_lc_f1(title='Adult Income: f1 Score/Training set size (kNN)', train_sizes=training_sizes_f1,
               train_scores_mean=train_scores_mean_kNN, validation_scores_mean=validation_scores_mean_kNN)

    '''Learning curve for Adult SVM'''
    from sklearn import svm
    classifier = svm.SVC(kernel='linear', class_weight={ 0:0.33, 1:0.66})

    train_sizes_SVM, train_scores_SVM, validation_scores_SVM = learning_curve(estimator=classifier,
                                                                              X=X1_min_norm, y=y1,
                                                                              train_sizes=training_sizes_f1,
                                                                              scoring='f1', cv=5)
    train_scores_mean_SVM = train_scores_SVM.mean(axis=1)
    validation_scores_mean_SVM = validation_scores_SVM.mean(axis=1)
    plot_lc_f1(title='Adult Income: f1 Score/Training set size (kNN)', train_sizes=training_sizes_f1,
               train_scores_mean=train_scores_mean_SVM, validation_scores_mean=validation_scores_mean_SVM)

    '''Learning curve for Adult decision Tree'''
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(max_depth=6)

    train_sizes_DT, train_scores_DT, validation_scores_DT = learning_curve(estimator=classifier,
                                                                           X=X2, y=y2,
                                                                           train_sizes=training_sizes_accuracy,
                                                                           scoring='accuracy', cv=5)
    train_scores_mean_DT = train_scores_DT.mean(axis=1)
    validation_scores_mean_DT = validation_scores_DT.mean(axis=1)
    plot_lc_f1(title='Phishing: f1 Score/Training set size (DT)', train_sizes=training_sizes_accuracy,
               train_scores_mean=train_scores_mean_DT, validation_scores_mean=validation_scores_mean_DT)

    '''Learning curve for Phishing Neural Network'''
    from sklearn.neural_network import MLPClassifier
    classifier = MLPClassifier((55, 55))

    train_sizes_NN, train_scores_NN, validation_scores_NN = learning_curve(estimator=classifier,
                                                                           X=X2, y=y2,
                                                                           train_sizes=training_sizes_accuracy,
                                                                           scoring='accuracy', cv=5)
    train_scores_mean_NN = train_scores_NN.mean(axis=1)
    validation_scores_mean_NN = validation_scores_NN.mean(axis=1)
    plot_lc_f1(title='Phishing: f1 Score/Training set size (NN)', train_sizes=training_sizes_accuracy,
               train_scores_mean=train_scores_mean_NN, validation_scores_mean=validation_scores_mean_NN)

    '''Learning curve for Phishing Boosted Decision Tree'''
    params = {'n_estimators': 50, 'max_depth': 5, 'min_samples_split': 2}
    from sklearn.ensemble import GradientBoostingClassifier
    classifier = GradientBoostingClassifier(**params)

    train_sizes_BDT, train_scores_BDT, validation_scores_BDT = learning_curve(estimator=classifier,
                                                                              X=X2, y=y2,
                                                                              train_sizes=training_sizes_accuracy,
                                                                              scoring='accuracy', cv=5)
    train_scores_mean_BDT = train_scores_BDT.mean(axis=1)
    validation_scores_mean_BDT = validation_scores_BDT.mean(axis=1)
    plot_lc_f1(title='Phishing: Accuracy/Training set size (BDT)', train_sizes=training_sizes_accuracy,
               train_scores_mean=train_scores_mean_BDT, validation_scores_mean=validation_scores_mean_BDT)

    '''Learning curve for Phishing KNN'''
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors=5)

    train_sizes_kNN, train_scores_kNN, validation_scores_kNN = learning_curve(estimator=classifier,
                                                                              X=X2, y=y2,
                                                                              train_sizes=training_sizes_accuracy,
                                                                              scoring='accuracy', cv=5)
    train_scores_mean_kNN = train_scores_kNN.mean(axis=1)
    validation_scores_mean_kNN = validation_scores_kNN.mean(axis=1)
    plot_lc_f1(title='Phishing: Accuracy/Training set size (kNN)', train_sizes=training_sizes_accuracy,
               train_scores_mean=train_scores_mean_kNN, validation_scores_mean=validation_scores_mean_kNN)

    '''Learning curve for Phishing SVM'''
    from sklearn import svm
    classifier = svm.SVC(kernel='linear')

    train_sizes_SVM, train_scores_SVM, validation_scores_SVM = learning_curve(estimator=classifier,
                                                                              X=X2, y=y2,
                                                                              train_sizes=training_sizes_accuracy,
                                                                              scoring='accuracy', cv=5)
    train_scores_mean_SVM = train_scores_SVM.mean(axis=1)
    validation_scores_mean_SVM = validation_scores_SVM.mean(axis=1)
    plot_lc_f1(title='Phishing: Accuracy/Training set size (SVM)', train_sizes=training_sizes_accuracy,
               train_scores_mean=train_scores_mean_SVM, validation_scores_mean=validation_scores_mean_SVM)
def decision_Tree_1(X_train, X_test, y_train, y_test):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    sc.fit(X_train)
    X_train = sc.transform(X_train)
    X_test = sc.transform(X_test)

    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(max_depth=9)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    from sklearn.metrics import classification_report, confusion_matrix
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

def neural_net_1(X_train, X_test, y_train, y_test):
    from sklearn.preprocessing import normalize
    from sklearn.preprocessing import minmax_scale
    X_train = minmax_scale(X_train)
    X_test = minmax_scale(X_test)
    X_train = normalize(X_train)
    X_test = normalize(X_test)

    from sklearn.neural_network import MLPClassifier

    neuralnet = MLPClassifier((95) * 2)
    neuralnet.fit(X_train, y_train)

    y_pred = neuralnet.predict(X_test)

    from sklearn.metrics import classification_report, confusion_matrix
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


def boosted_DT_1(X_train, X_test, y_train, y_test):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    sc.fit(X_train)
    X_train = sc.transform(X_train)
    X_test = sc.transform(X_test)

    from sklearn.ensemble import GradientBoostingClassifier
    params = {'n_estimators': 50, 'max_depth': 9, 'min_samples_split': 2}
    boostyboi = GradientBoostingClassifier(**params)
    boostyboi.fit(X_train, y_train)

    y_pred = boostyboi.predict(X_test)
    from sklearn.metrics import classification_report, confusion_matrix
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

def kNN_1(X_train, X_test, y_train, y_test):
    from sklearn.preprocessing import normalize
    from sklearn.preprocessing import minmax_scale
    X_train = minmax_scale(X_train)
    X_test = minmax_scale(X_test)
    X_train = normalize(X_train)
    X_test = normalize(X_test)

    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors=13)

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    from sklearn.metrics import classification_report, confusion_matrix
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

def SVM_1(X_train, X_test, y_train, y_test):
    from sklearn.preprocessing import normalize
    from sklearn.preprocessing import minmax_scale
    X_train = minmax_scale(X_train)
    X_test = minmax_scale(X_test)
    X_train = normalize(X_train)
    X_test = normalize(X_test)

    from sklearn import svm

    classifier = svm.SVC(kernel='linear', class_weight={ 0:0.33, 1:0.66})
    classifier.fit(X_train,y_train)

    y_pred = classifier.predict(X_test)
    from sklearn.metrics import classification_report, confusion_matrix
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

def decision_Tree_2(X_train, X_test, y_train, y_test):

    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(max_depth=6)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    from sklearn.metrics import classification_report, confusion_matrix
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


def neural_net_2(X_train, X_test, y_train, y_test):
    from sklearn.neural_network import MLPClassifier

    neuralnet = MLPClassifier((55)*2)
    neuralnet.fit(X_train,y_train)

    y_pred = neuralnet.predict(X_test)

    from sklearn.metrics import classification_report, confusion_matrix
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

def boosted_DT_2(X_train, X_test, y_train, y_test):

    from sklearn.ensemble import GradientBoostingClassifier
    params = {'n_estimators': 50, 'max_depth': 5, 'min_samples_split': 2}
    boostyboi = GradientBoostingClassifier(**params)
    boostyboi.fit(X_train, y_train)

    y_pred = boostyboi.predict(X_test)
    from sklearn.metrics import classification_report, confusion_matrix
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

def kNN_2(X_train, X_test, y_train, y_test):

    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors=13)

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    from sklearn.metrics import classification_report, confusion_matrix
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

def SVM_2(X_train, X_test, y_train, y_test):
    from sklearn import svm

    classifier = svm.SVC(kernel='linear')
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    from sklearn.metrics import classification_report, confusion_matrix
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


def plot_lc_accuracy(title, train_sizes, train_scores_mean, validation_scores_mean):
        plot.figure()
        plot.title(title)
        plot.plot(train_sizes, validation_scores_mean, '-', label='Validation Accuracy (Test)')
        plot.plot(train_sizes, train_scores_mean, '-', label='Training Accuracy')
        plot.legend()
        plot.xlabel('Training Set Size')
        plot.ylabel('Accuracy')
        plot.show()


def plot_lc_f1(title, train_sizes, train_scores_mean, validation_scores_mean):
        plot.figure()
        plot.title(title)
        plot.plot(train_sizes, validation_scores_mean, '-', label='Validation f1 score (Test)')
        plot.plot(train_sizes, train_scores_mean, '-', label='Training f1 score')
        plot.legend()
        plot.xlabel('Training Set Size')
        plot.ylabel('f1 Score')
        plot.show()

main()
