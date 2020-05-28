from pyTsetlinMachineParallel.tm import MultiClassConvolutionalTsetlinMachine2D
import numpy as np
from time import time
import math

X_train = np.array([])
Y_train = np.array([])
X_test = np.array([])
Y_test = np.array([])

# base_path_start = "Data/KfoldDataStaticTransformed/"
base_path_start = ""
base_path_end = "statickfold.data"


# path_train = "Data/eventrain.data"
# path_test = "Data/eventest.data"


def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier


def tsetlinConvolutionalWeighted(epochs, clauses, T, s, k_fold_amount, shape_x, shape_y, shape_z, frame_x, frame_y):
    print("epochs = ", epochs)
    print("clauses = ", clauses)
    print("T = ", T)
    print("s = ", s, "\n")

    return mergingKFold(k_fold_amount, clauses, T, s, epochs, shape_x, shape_y, shape_z, frame_x, frame_y)


def tsetlinConvolutionalWeightedHandler(epochs, clauses, T, s, k_fold_amount, shape_x, shape_y, shape_z, frame_x, frame_y):
    # Shape of the game board
    print("shape_x = ", shape_x)
    print("shape_y = ", shape_y)
    print("shape_z = ", shape_z, "\n")

    # Shape of the window for ConvTM moving around in the game board
    print("frame_x = ", frame_x)
    print("frame_y = ", frame_y, "\n")

    score = tsetlinConvolutionalWeighted(epochs, clauses, T, s, k_fold_amount, shape_x, shape_y, shape_z, frame_x, frame_y)
    print(score)
    mean_score = round(sum(score) / len(score), 2)

    with open("ConvolutionalTsetlinWeightedLogPositiveBoost.txt", "a+") as myfile:
        myfile.write("Clauses: " + str(clauses) + ", Treshold: " + str(T) + ", S: " + str(s) + ", Epochs: " + str(
        epochs) + ", Shape x: " + str(shape_x) + ", Shape y: " + str(shape_y) + ", Shape z: " + str(shape_z) + ", Frame x: " + str(frame_x) + ", Frame y: " + str(frame_y) + ", Mean accuracy: " + str(mean_score) + ", Score per Kfold: " + str(score) + "\n")
    myfile.close()

    print("Clauses: " + str(clauses) + ", Treshold: " + str(T) + ", S: " + str(s) + ", Epochs: " + str(
        epochs) + ", Shape x: " + str(shape_x) + ", Shape y: " + str(shape_y) + ", Shape z: " + str(shape_z) + ", Frame x: " + str(frame_x) + ", Frame y: " + str(frame_y) + ", Mean accuracy: " + str(mean_score) + ", Score per Kfold: " + str(score) + "\n")
    print("Results saved to file")


def mergingKFold(file_amount, _clauses, _T, _s, _epochs, shape_x, shape_y, shape_z, frame_x, frame_y):
    results = []
    for i in range(file_amount):
        print("")
        print("Starting Kfold number ", i + 1)
        train_string = base_path_start + str(i) + "train" + base_path_end
        test_string = base_path_start + str(i) + "test" + base_path_end
        score = loadingData(train_string, test_string, _clauses, _T, _s, _epochs, shape_x, shape_y, shape_z, frame_x, frame_y)
        score = round_up(score, 2)
        results.append(score)

    return results



# shape[0] = length of dataset.
# shape[1] | shape_x = length of x-axis
# shape[2] | shape_y = length of y-axis
# shape[3] | shape_z = length of z-axis(if 3D)
def loadingData(_train, _test, _clauses, _T, _s, _epochs, shape_x, shape_y, shape_z, frame_x, frame_y):
    print("Loading training data..")
    train_data = np.loadtxt(_train, delimiter=",")
    global X_train
    global Y_train
    X_train = train_data[:, 0:-1].reshape(train_data.shape[0], shape_x, shape_y, shape_z)
    Y_train = train_data[:, -1]
    print("X_train.shape[0]: ", X_train.shape[0])
    print("X_train.shape[1]: ", X_train.shape[1])
    print("X_train.shape[2]: ", X_train.shape[2])
    print("X_train.shape[3]: ", X_train.shape[3], "\n")

    print("Loading test data..")
    test_data = np.loadtxt(_test, delimiter=",")
    global X_test
    global Y_test
    X_test = test_data[:, 0:-1].reshape(test_data.shape[0], shape_x, shape_y, shape_z)
    Y_test = test_data[:, -1]
    print("X_test.shape[0]: ", X_test.shape[0])
    print("X_test.shape[1]: ", X_test.shape[1])
    print("X_test.shape[2]: ", X_test.shape[2])
    print("X_test.shape[3]: ", X_test.shape[3], "\n")

    return ConvTM(_clauses, _T, _s, _epochs, frame_x, frame_y)







def ConvTM(_clauses, _T, _s, _epochs, _frame_x, _frame_y):
    print("Creating MultiClass Convolutional Tsetlin Machine.")
    tm = MultiClassConvolutionalTsetlinMachine2D(_clauses, _T, _s, (_frame_x, _frame_y), boost_true_positive_feedback=True)
    print("Starting ConvTM with weighted clauses..")
    print("\nAccuracy over " + str(_epochs) + " epochs:\n")
    past_old_10_epochs = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    past_10_epochs = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    for i in range(_epochs):
        start = time()
        tm.fit(X_train, Y_train, epochs=1, incremental=True)
        stop = time()
        result = 100 * (tm.predict(X_test) == Y_test).mean()
        print("#%d Accuracy: %.2f%% (%.2fs)" % (i + 1, result, stop - start))

        past_old_10_epochs.pop(0)
        past_old_10_epochs.append(past_10_epochs[0])
        past_10_epochs.pop(0)
        past_10_epochs.append(round_up(result, 2))

        long_period_average_score = round(sum(past_old_10_epochs) / len(past_old_10_epochs), 2)
        this_period_average_score = round(sum(past_10_epochs) / len(past_10_epochs))
        if (long_period_average_score > this_period_average_score):
            print("long_period_average_score: ", long_period_average_score, " vs ", "this_period_average_score: ",
                  this_period_average_score)
            break

    mean_accuracy = 100 * (tm.predict(X_test) == Y_test).mean()
    print("Mean Accuracy:", mean_accuracy, "\n")
    print("Finished running..")

    return mean_accuracy




# Parameters
# split_ratio = 0.9

k_fold_amount = 10

# setlinConvolutionalWeightedHandler(epochs, clauses, T, s, k_fold_amount, shape_x, shape_y, shape_z, frame_x, frame_y):

tsetlinConvolutionalWeightedHandler(500, 10000, 120000, 30, k_fold_amount, 4, 8, 4, 2, 2)


"""
# Shape of the game board
shape_x = 7
shape_y = 6
shape_z = 2
print("shape_x = ", shape_x)
print("shape_y = ", shape_y)
print("shape_z = ", shape_z, "\n")

# Shape of the window for ConvTM moving around in the game board
frame_x = 4
frame_y = 4
print("frame_x = ", frame_x)
print("frame_y = ", frame_y, "\n")
"""