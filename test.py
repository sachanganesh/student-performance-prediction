"""

The goal of this experiment is to develop a model that will predict (to a certain level of acceptable accuracy) the cumulative grades of students learning mathematics over a year (two semesters).
The features will consist of all given features within the dataset excluding the GX factors that represent semester grades. G3 will be the feature to be predicted.

"""

""" Import helper libraries """
import numpy as np
import pandas as pd

""" Read data file as DataFrame """
df = pd.read_csv("./data/student-mat.csv", sep=";")

""" Import ML helpers """
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import LinearSVC # Support Vector Machine Classifier model

def split_data(X, Y):
    return train_test_split(X, Y, test_size=0.3, random_state=17)

""" Confusion Matrix """
def confuse(y_true, y_pred):
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    print("\nConfusion Matrix: \n", cm)
    fpr(cm)

""" False Positive Rate """
def fpr(confusion_matrix):
    fp = confusion_matrix[0][1]
    tn = confusion_matrix[0][0]
    rate = float(fp) / (fp + tn)
    print("\nFalse Positive Rate: ", rate)

    return rate

def main():
    # For each nominal feature, encode to ordinal values
    class_le = LabelEncoder()
    for column in df[["school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason", "guardian", "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic"]].columns:
        df[column] = class_le.fit_transform(df[column].values)

    """ Remove periodic grade features """
    df.drop(["G1", "G2"], axis = 1, inplace=True)
    # df.drop(["G1", "G2", "romantic", "Dalc", "sex", "traveltime", "paid", "activities", "nursery", "famsup", "address", "famsize", "schoolsup", "internet", "higher", "school", "Pstatus"], axis=1, inplace=True)

    # Encode G3 as pass or fail binary values
    for i, row in df.iterrows():
        if row["G3"] >= 10:
            df["G3"][i] = 1
        else:
            df["G3"][i] = 0

    df_orig = df.copy(deep=True)

    # Target values are G3
    y = df.pop("G3")

    # Feature set is remaining features
    X = df

    print("\nStudent Performance Predictive Modeling")

    ##############################################################################################

    X_train, X_test, y_train, y_test = split_data(X, y)

    clf = Pipeline([
        ('reduce_dim', SelectKBest(chi2, k=2)),
        ('train', LinearSVC(C=100))
    ])

    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))

    confuse(y_test, clf.predict(X_test))

    # params = [
    #     {
    #         "kernel": ["rbf", "linear"],
    #         "C": [1, 10, 100, 1000]
    #     }
    # ]
    #
    # clf = GridSearchCV(SVC(C=1), params, cv=5)
    # clf.fit(X_train, y_train)
    #
    # k=10, linear, 100: 0.721 mean, 0.118 std
    # school, absences, reason, schoolsup

main()
