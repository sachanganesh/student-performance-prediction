"""

The goal of this experiment is to develop a model that will predict (to a certain level of acceptable accuracy) the cumulative grades of students learning mathematics over a year (two semesters).
The features will consist of all given features within the dataset excluding the GX factors that represent semester grades. G3 will be the feature to be predicted.

"""

""" Import helper libraries """
import numpy as np
import pandas as pd

""" Read data file as DataFrame """
df = pd.read_csv("./data/student-mat.csv", sep=";")

""" Perform preprocessing of data """
# Import LabelEncoder lib to transform nominal features
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def split_data(X, Y):
    return train_test_split(X, Y, test_size=0.3, random_state=0)

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

""" Principle Component Analysis """
def principle_component_analysis(X, n_components):
    stdsc = StandardScaler()
    X_std = stdsc.fit_transform(X)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_std)

    return X_pca, pca

""" Kernel SVM Classifier """
def svm(X_train, Y_train, X_test, Y_test):
    svc = SVC(kernel="rbf", C=10)
    svc.fit(X_train, Y_train)
    print("\n\nSVC Accuracy: ", svc.score(X_test, Y_test))
    confuse(Y_test, svc.predict(X_test))

    return svc

""" Gaussian Naive Bayes Binary Classifier """
def biased_naive_bayes(df, X_test, Y_test):
    fail_df = df.copy(deep=True).loc[df["G3"] == 0]
    pass_df = df.copy(deep=True).loc[df["G3"] == 1]

    # Target values are G3
    Y = df.pop("G3")
    Y_fail = fail_df.pop("G3")
    Y_pass = pass_df.pop("G3")

    # Feature set is remaining features
    X = df
    X_fail = fail_df
    X_pass = pass_df

    gnb = GaussianNB()
    for i in (0, 3):
        gnb.partial_fit(X_fail, Y_fail, [0, 1])
    gnb.partial_fit(X_pass, Y_pass, [0, 1])
    for i in (0, 3):
        gnb.partial_fit(X_fail, Y_fail, [0, 1])

    print("\n\nGuassian Naive Bayes (Boosted) Accuracy: ", gnb.score(X_test, Y_test))
    confuse(Y, gnb.predict(X))

    return gnb

def naive_bayes(X_train, Y_train, X_test, Y_test):
    gnb = GaussianNB()
    gnb.fit(X_train, Y_train)

    print("\n\nGaussian Naive Bayes Accuracy: ", gnb.score(X_test, Y_test))
    confuse(Y_test, gnb.predict(X_test))

    return gnb

""" Logistic Regression Classifier """
def log_regression(X_train, Y_train, X_test, Y_test):
    lrc = LogisticRegression(penalty="l1")
    lrc.fit(X_train, Y_train)
    print("\n\nLogistic Regression Accuracy: ",  lrc.score(X_test, Y_test))
    confuse(Y_test, lrc.predict(X_test))

    return lrc

def main():
    # For each nominal feature, encode to ordinal values
    class_le = LabelEncoder()
    for column in df[["school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason", "guardian", "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic"]].columns:
        df[column] = class_le.fit_transform(df[column].values)

    """ Remove periodic grade features """
    df.drop(["G1", "G2"], axis = 1, inplace = True)
    # df.drop(["G1", "G2", "romantic", "Dalc", "sex", "traveltime", "paid", "activities", "nursery", "famsup", "address", "famsize", "schoolsup", "internet", "higher", "school", "Pstatus"], axis=1, inplace=True)

    # Encode G3 as pass or fail binary values
    for i, row in df.iterrows():
        if row["G3"] >= 10:
            df["G3"][i] = 1
        else:
            df["G3"][i] = 0

    df_orig = df.copy(deep=True)

    # Target values are G3
    Y = df.pop("G3")

    # Feature set is remaining features
    X = df

    # Scale the feature set
    # stdsc = StandardScaler()
    # X_train_std = stdsc.fit_transform(X_train)
    # X_test_std = stdsc.transform(X_test)

    print("\nStudent Performance Predictive Modeling")

    ##############################################################################################
    X_pca, pca = principle_component_analysis(X, n_components=11)

    # Split data into training and testing sets
    X_train, X_test, Y_train, Y_test = split_data(X, Y)

    print("\nPCA Ratios: ", pca.explained_variance_ratio_)

    # Attempt to split data before boosting for fair evaluation
    X_train = X_train.assign(G3 = Y_train.values)
    biased_naive_bayes(X_train, X_test, Y_test)
    
    clf = naive_bayes(X_train, Y_train, X_test, Y_test)
    joblib.dump(clf, "./models/latest_build.pkl")
   
    """ Present Best Models
    m1 = joblib.load("./models/best_classic.pkl")
    print("\nModel 1: ", m1.score(X, Y))
    confuse(Y, m1.predict(X))

    m2 = joblib.load("./models/best_match.pkl")
    print("\nModel 2: ", m2.score(X, Y))
    confuse(Y, m2.predict(X))

    m3 = joblib.load("./models/best_fpr.pkl")
    print("\nModel 3: ", m3.score(X, Y))
    confuse(Y, m3.predict(X))
    """

main()
