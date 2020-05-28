#https://realpython.com/logistic-regression-python/
from time import time
import math
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

X_train = np.array([])
Y_train = np.array([])
X_test = np.array([])
Y_test = np.array([])

base_path_start = ""
base_path_end = "statickfold.data"


def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier

def mergingKFold(file_amount, solver, random_state, max_iter):
    results = []
    for i in range(file_amount):
        print("")
        print("Starting Kfold number ", i+1)
        train_string = base_path_start + str(i) + "train" + base_path_end
        test_string = base_path_start + str(i) + "test" + base_path_end
        score = loadingData(train_string, test_string, solver, random_state, max_iter)
        score = round_up(score, 2)
        results.append(score)

    return results


def loadingData(_train, _test, solver, random_state, max_iter):
    print("Loading training data..")
    train_data = np.loadtxt(_train, delimiter=",")
    # print("..using train dataset: ", _path_train)
    #global X_train
    #global Y_train
    X_train = train_data[:, 0:-1]
    Y_train = train_data[:, -1]

    print("Loading test data..")
    test_data = np.loadtxt(_test, delimiter=",")
    # print("..using test dataset: ", _path_test)
    #global X_test
    #global Y_test
    X_test = test_data[:, 0:-1]
    Y_test = test_data[:, -1]

    model = LogisticRegression(solver=solver, random_state=random_state, max_iter=max_iter).fit(X_train, Y_train)
    accuracy = model.score(X_test, Y_test)
    print(accuracy)
    return accuracy




def LRegression(k_fold_amount, solver, random_state, max_iter):
    print("Settings:")
    print("solver " + str(solver) + ", random_state " + str(random_state) + ", max_iter " + str(max_iter))
    score = mergingKFold(k_fold_amount, solver, random_state, max_iter)
    print(score)
    mean_score = round(sum(score) / len(score), 2)

    # acstr = str(accuracy)
    # with open("StandardTsetlinWeightedLog.txt", "a+") as myfile:
    with open("LogisticRegressionLog.txt", "a+") as myfile:
        myfile.write("Mean accuracy: " + str(mean_score) + ", Score per Kfold: " + str(score) + ", Settings: " + "solver " + str(solver) + ", random_state " + str(random_state) + ", max_iter " + str(max_iter) + "\n")
    myfile.close()

    print("Mean accuracy: " + str(mean_score) + ", Score per Kfold: " + str(score) + ", Settings: " + "solver " + str(solver) + ", random_state " + str(random_state) + ", max_iter " + str(max_iter) + "\n")
    print("Results saved to file")



LRegression(10, "liblinear", 0, 1000)
LRegression(10, "newton-cg", 0, 1000)
LRegression(10, "lbfgs", 0, 1000)
LRegression(10, "sag", 0, 1000)
LRegression(10, "saga", 0, 1000)
LRegression(10, "liblinear", 10, 1000)
LRegression(10, "newton-cg", 10, 1000)
LRegression(10, "lbfgs", 10, 1000)
LRegression(10, "sag", 10, 1000)
LRegression(10, "saga", 10, 1000)
LRegression(10, "liblinear", 32, 1000)
LRegression(10, "newton-cg", 32, 1000)
LRegression(10, "lbfgs", 32, 1000)
LRegression(10, "sag", 32, 1000)
LRegression(10, "saga", 32, 1000)
LRegression(10, "liblinear", 45, 1000)
LRegression(10, "newton-cg", 45, 1000)
LRegression(10, "lbfgs", 45, 1000)
LRegression(10, "sag", 45, 1000)
LRegression(10, "saga", 45, 1000)
LRegression(10, "liblinear", 61, 1000)
LRegression(10, "newton-cg", 61, 1000)
LRegression(10, "lbfgs", 61, 1000)
LRegression(10, "sag", 61, 1000)
LRegression(10, "saga", 61, 1000)

