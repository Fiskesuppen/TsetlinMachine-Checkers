from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
import numpy as np
from time import time
import math

X_train = np.array([])
Y_train = np.array([])
X_test = np.array([])
Y_test = np.array([])

#base_path_start = "Data/KfoldDataStaticTransformed/"
base_path_start = "data/"
base_path_end = "statickfoldblack.data"
# path_train = "Data/eventrain.data"
# path_test = "Data/eventest.data"


def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier


def tsetlinStandardWeighted(epochs, clauses, T, s):
    print("epochs = ", epochs)
    print("clauses = ", clauses)
    print("T = ", T)
    print("s = ", s, "\n")

    #decide what train and test data to use: str(number)
    train_string = base_path_start + str(2) + "train" + base_path_end
    test_string = base_path_start + str(2) + "test" + base_path_end
    
    print("Loading training data..")
    train_data = np.loadtxt(train_string, delimiter=",")
    # print("..using train dataset: ", _path_train)
    global X_train
    global Y_train
    X_train = train_data[:, 0:-1]
    Y_train = train_data[:, -1]
    
    print("Loading test data..")
    test_data = np.loadtxt(test_string, delimiter=",")
    # print("..using test dataset: ", _path_test)
    global X_test
    global Y_test
    X_test = test_data[:, 0:-1]
    Y_test = test_data[:, -1]
    
    
    return TM(clauses, T, s, epochs)


def tsetlinStandardWeightedHandler(epochs, clauses, T, s):

    
    score = tsetlinStandardWeighted(epochs, clauses, T, s)
    
    print(score)

    #acstr = str(accuracy)
    #with open("StandardTsetlinWeightedLog.txt", "a+") as myfile:
    with open("TrainTsetlinSaveStateBlackLog.txt", "a+") as myfile:
        myfile.write("Clauses: " + str(clauses) + ", Treshold: " + str(T) + ", S: " + str(s) + ", Epochs: " + str(epochs) + ", Accuracy: " + str(score) + "\n")
    myfile.close()

    print("Clauses: " + str(clauses) + ", Treshold: " + str(T) + ", S: " + str(s) + ", Epochs: " + str(epochs) + ", Accuracy: " + str(score) + "\n")
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



def SaveStates(t_machine):
    print("Saving states")
    try:
        statefilename = "SavedStatesBlack"
        t_machine.fit(X_train, Y_train, epochs=0, incremental=True)
        np.save(statefilename, t_machine.get_state())
        print("States succesfully saved in file: ", statefilename)
    except FileNotFoundError:
        print("Could not save states, file or directory not found.")
        exit(22)

"""        
def LoadStates(t_machine):
        try:
            t_states = np.load("SavedStates.npy", allow_pickle=True)
        except FileNotFoundError:
            print("Could not loadk state, file or directory not found.")
            exit(33)
        t_machine.fit(X_train, Y_train, epochs=1, incremental=True)
        t_machine.set_state(t_states)
        return t_machine
"""
            
            
def load_tm_state(_m, _x_train, _y_train, _start_epoch, _clauses, _t, _s, _window_x, _window_y, _shape_x, _shape_y,
                      _shape_z):
        global load_state
        _start_epoch = 1
        try:
            _tm_state = np.load(load_path + str(counter) + ".npy", allow_pickle=True)
        except FileNotFoundError:
            print("Could not load TM state. File or directory not found.")
            load_state = False
            return _start_epoch
        _m.fit(_x_train, _y_train, epochs=0, incremental=True)
        _m.set_state(_tm_state)
        loaded_results_list = load_results(_clauses, _t, _s, _window_x, _window_y, _shape_x, _shape_y, _shape_z)
        _start_epoch = set_results(_start_epoch, loaded_results_list)

        return _start_epoch
 
        
    
def TM(_clauses, _T, _s, _epochs):
    print("Creating MultiClass Tsetlin Machine.")
    tm = MultiClassTsetlinMachine(_clauses, _T, _s, boost_true_positive_feedback=1, weighted_clauses=True)
    print("Starting TM with weighted clauses..")
    print("\nAccuracy over ", _epochs, " epochs:\n")
    past_old_10_epochs = [10,10,10,10,10,10,10,10,10,10,10]
    past_10_epochs = [10,10,10,10,10,10,10,10,10,10]
    for i in range(_epochs):
        start = time()
        tm.fit(X_train, Y_train, epochs=1, incremental=True)
        stop = time()
        result = 100 * (tm.predict(X_test) == Y_test).mean()
        print("#%d Accuracy: %.2f%% (%.2fs)" % (i + 1, result, stop - start))
        
        past_old_10_epochs.pop(0)
        past_old_10_epochs.append(past_10_epochs[0])
        past_10_epochs.pop(0)
        past_10_epochs.append(round_up(result,2))
        
        
        long_period_average_score = round(sum(past_old_10_epochs)/len(past_old_10_epochs),2)
        this_period_average_score= round(sum(past_10_epochs)/len(past_10_epochs))
        if(long_period_average_score > this_period_average_score):
            print("long_period_average_score: ", long_period_average_score, " vs ", "this_period_average_score: ", this_period_average_score)
            break

    mean_accuracy = 100 * (tm.predict(X_test) == Y_test).mean()
    print("Most Recent Accuracy:", mean_accuracy)
    print("Finished running.. \n")
    
    SaveStates(tm)
    
    return mean_accuracy


# Parameters
# split_ratio = 0.9


#tsetlinStandardWeightedHandler(epochs, clauses, T, s)



tsetlinStandardWeightedHandler(500, 19000, 40000, 9)
#This one also uses data slice number 3

#tsetlinStandardWeightedHandler(500, 100, 28000, 40)



