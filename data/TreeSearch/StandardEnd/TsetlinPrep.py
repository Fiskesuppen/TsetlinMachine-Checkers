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

    train_string = base_path_start + str(0) + "train" + base_path_end
    test_string = base_path_start + str(0) + "test" + base_path_end
    
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
    TM = tsetlinStandardWeighted(epochs, clauses, T, s)
    print("TM successfully prepared")
    return TM



        
def LoadStates(t_machine):
        print("Loading states")
        try:
            t_states = np.load(statefile, allow_pickle=True)
        except FileNotFoundError:
            print("Could not load state, file or directory not found.")
            exit(33)
        # Make sure X_train and Y_train is the same as the one used during training
        t_machine.fit(X_train, Y_train, epochs=1, incremental=True)
        t_machine.set_state(t_states)
        print("TM states successfully loaded from file: ", statefile)
        return t_machine
            
            
            

        
    
def TM(_clauses, _T, _s, _epochs):
    print("Creating MultiClass Tsetlin Machine.")
    tm = MultiClassTsetlinMachine(_clauses, _T, _s, boost_true_positive_feedback=1, weighted_clauses=True)
    tm = LoadStates(tm)
    result = round (100 * (tm.predict(X_test) == Y_test).mean(),2)
    print("Trained accuracy: ", result)
    return tm

def TMInitWhite(epochs, clauses, treshold, s):
    print("Initializing White Tsetlin")
    global base_path_end
    base_path_end = "statickfoldwhite.data"
    global statefile
    statefile = "SavedStatesWhite.npy"
    global tmwhite
    tmwhite = tsetlinStandardWeightedHandler(epochs, clauses, treshold, s)

def TMInitBlack(epochs, clauses, treshold, s):
    print("Initializing Black Tsetlin")
    global base_path_end
    base_path_end = "statickfoldblack.data"
    global statefile
    statefile = "SavedStatesBlack.npy"
    global tmblack
    tmblack = tsetlinStandardWeightedHandler(epochs, clauses, treshold, s)
    
    
def toBinary(board):
    """
    1 = black pawn
    2 = black king
    3 = white pawn
    4 = white king
    """
    if (len(board) != 32):
        print("Error in length of board, not equal to 32")
    binary_board = []
    for q in range(len(board)):
        if (board[q] == 1):
            binary_board.append(1)
        else:
            binary_board.append(0)
    for q in range(len(board)):
        if (board[q] == 2):
            binary_board.append(1)
        else:
            binary_board.append(0)
    for q in range(len(board)):
        if (board[q] == 3):
            binary_board.append(1)
        else:
            binary_board.append(0)
    for q in range(len(board)):
        if (board[q] == 4):
            binary_board.append(1)
        else:
            binary_board.append(0)

    return binary_board


#Evaluates the board, returns loss 0, win 1 or draw 0.5
def evaluate(board, tm):
    #tm is either tmblack or tmwhite. tmblack evaluates the winning chance of black player. tmwhite evaluates the winning chance of white player.
    board = toBinary(board)
    check_board = []
    check_board.append(board)
    check_board = np.asarray(check_board)
    result = tm.predict(check_board)
    #Draw returns 0.5, a value or score between loss and win.
    #Draw
    if (result  == 2):
        return 0.5
    #Win
    elif (result == 1):
        return 1
    #Loss
    elif (result == 0):
        return 0
    else:
        print("Result in evaluate not 2, 1 nor 0. This should not happen")
        exit(69)

#Evaluates the board, returns loss 0, win 1 or draw 0.5
def multiEvaluate(board, tm):
    #tm is either tmblack or tmwhite. tmblack evaluates the winning chance of black player. tmwhite evaluates the winning chance of white player.
    #board = toBinary(board)
    check_board = np.asarray(board)
    result = tm.predict(check_board)
    #Draw returns 0.5, a value or score between loss and win.
    #Draw
    results = []
    for i in range(len(result)):
        if (result[i]  == 2):
            results.append(0.5)
        #Win
        elif (result[i] == 1):
            results.append(1)
        #Loss
        elif (result[i] == 0):
            results.append(0)
        else:
            print("Result in evaluate not 2, 1 nor 0. This should not happen")
            exit(69)
    return results







#tsetlinStandardWeightedHandler(epochs, clauses, T, s)
#tsetlinStandardWeightedHandler(500, 15000, 28000, 15)

# Make sure the settings are equal to those used when training
TMInitBlack(500, 15000, 28000, 15)
print(" ")
print(" ")
TMInitWhite(500, 15000, 28000, 15)

#Clear the Tsetlin initializing data to relieve the memory a little.
X_train = 0
Y_train = 0
X_test = 0
Y_test = 0

"""
board = [4,0,1,0,0,0,0,3,1,0,0,3,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,0,0,3,3]
board = [1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,3,3,3,3,3,3,3,3,3,3,3,3]
result = evaluate(board, tmblack)
"""

#Evaluate whether it would be faster to have Tsetlin evaluate in bulk or not

#start_time = time.time()
#elapsed_time = time.time() - start_time


print(" ")

board = []
for i in range(2):
    board.append([1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,3,3,3,3,3,3,3,3,3,3,3,3])
    board.append([4,0,1,0,0,0,0,3,1,0,0,3,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,0,0,3,3])
    
    
start_time = time()

results = []
for i in range(len(board)):
    results.append(evaluate(board[i], tmblack))
    
stop_time = time()
print(len(results))
print("One by one time: ", stop_time - start_time)


start_time = time()

bin_boards = []
for i in range(len(board)):
    bin_boards.append(toBinary(board[i]))
results = multiEvaluate(bin_boards, tmblack)

stop_time = time()
print(len(results))
print("Bulk by bulk time: ", stop_time - start_time)
