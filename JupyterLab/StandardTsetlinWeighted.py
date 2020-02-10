from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
import numpy as np
from time import time
import math

X_train = np.array([])
Y_train = np.array([])
X_test = np.array([])
Y_test = np.array([])

base_path_start = "Data/KfoldDataStaticTransformed/"
base_path_end = "statickfold.data"
# path_train = "Data/eventrain.data"
# path_test = "Data/eventest.data"


def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier


def tsetlinStandardWeighted(epochs, clauses, T, s, k_fold_amount):
    print("epochs = ", epochs)
    print("clauses = ", clauses)
    print("T = ", T)
    print("s = ", s, "\n")

    return mergingKFold(k_fold_amount, clauses, T, s, epochs)


def tsetlinStandardWeightedHandler(epochs, clauses, T, s, k_fold_amount):
    score = tsetlinStandardWeighted(k_fold_amount, clauses, T, s, epochs)
    print(score)
    average_score = round(sum(score)/len(score),2)

    #acstr = str(accuracy)
    with open("StandardTsetlinWeightedLog.txt", "a+") as myfile:
        myfile.write("Clauses: " + str(clauses) + ", Treshold: " + str(T) + ", S: " + str(s) + ", Epochs: " + str(epochs) + ", Mean accuracy: " + str(average_score) + ", Score per Kfold: " + str(score) + "\n")
    myfile.close()

    print("Clauses: " + str(clauses) + ", Treshold: " + str(T) + ", S: " + str(s) + ", Epochs: " + str(epochs) + ", Mean accuracy: " + str(average_score) + ", Score per Kfold: " + str(score) + "\n")
    print("Results saved to file")


def mergingKFold(file_amount, _clauses, _T, _s, _epochs):
    results = []
    for i in range(file_amount):
        print("")
        print("Starting Kfold number ", i+1)
        train_string = base_path_start + str(i) + "train" + base_path_end
        test_string = base_path_start + str(i) + "test" + base_path_end
        score = loadingData(train_string, test_string, _clauses, _T, _s, _epochs)
        score = round_up(score, 2)
        results.append(score)

    return results


def loadingData(_train, _test, _clauses, _T, _s, _epochs):
    print("Loading training data..")
    train_data = np.loadtxt(_train, delimiter=",")
    # print("..using train dataset: ", _path_train)
    global X_train
    global Y_train
    X_train = train_data[:, 0:-1]
    Y_train = train_data[:, -1]

    print("Loading test data..")
    test_data = np.loadtxt(_test, delimiter=",")
    # print("..using test dataset: ", _path_test)
    global X_test
    global Y_test
    X_test = test_data[:, 0:-1]
    Y_test = test_data[:, -1]

    return TM(_clauses, _T, _s, _epochs)


def TM(_clauses, _T, _s, _epochs):
    print("Creating MultiClass Tsetlin Machine.")
    tm = MultiClassTsetlinMachine(_clauses, _T, _s, boost_true_positive_feedback=0, weighted_clauses=True)
    print("Starting TM with weighted clauses..")
    print("\nAccuracy over ", _epochs, " epochs:\n")
    past_epochs = [10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10]
    past_10_epochs = [10,10,10,10,10,10,10,10,10,10]
    for i in range(_epochs):
        start = time()
        tm.fit(X_train, Y_train, epochs=1, incremental=True)
        stop = time()
        result = 100 * (tm.predict(X_test) == Y_test).mean()
        print("#%d Accuracy: %.2f%% (%.2fs)" % (i + 1, result, stop - start))
        past_epochs.pop(0)
        past_epochs.append(result)
        long_period_average_score = round(sum(past_epochs)/len(past_epochs),2)
        past_10_epochs.pop(0)
        past_10_epochs.append(result)
        this_period_average_score= round(sum(past_10_epochs)/len(past_10_epochs))
        if(long_period_average_score > this_period_average_score):
            print("long_period_average_score: ", long_period_average_score, " vs ", "this_period_average_score: ", this_period_average_score)
            break

    mean_accuracy = 100 * (tm.predict(X_test) == Y_test).mean()
    print("Mean Accuracy:", mean_accuracy)
    print("Finished running.. \n")

    return mean_accuracy


# Parameters
# split_ratio = 0.9

k_fold_amount = 10
#tsetlinStandardWeightedHandler(k_fold_amount, clauses, T, s, epochs)

tsetlinStandardWeightedHandler(k_fold_amount, 2000, 12000, 15, 500)
tsetlinStandardWeightedHandler(k_fold_amount, 3000, 12000, 15, 500)
tsetlinStandardWeightedHandler(k_fold_amount, 5000, 12000, 15, 500)
tsetlinStandardWeightedHandler(k_fold_amount, 8000, 12000, 15, 500)
tsetlinStandardWeightedHandler(k_fold_amount, 12000, 12000, 15, 500)
tsetlinStandardWeightedHandler(k_fold_amount, 15000, 12000, 15, 500)
