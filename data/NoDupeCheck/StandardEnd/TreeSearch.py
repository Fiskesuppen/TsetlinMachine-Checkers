from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
import numpy as np
from time import time
import math

X_train = np.array([])
Y_train = np.array([])
X_test = np.array([])
Y_test = np.array([])



#base_path_start = "Data/KfoldDataStaticTransformed/"
base_path_start = ""
base_path_end = "statickfold.data"
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
        statefile = "SavedStates.npy"
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

#Evaluates the board, returns loss 0, win 1 or draw 2
def evaluate(board):
    board = toBinary(board)
    check_board = []
    check_board.append(board)
    check_board = np.asarray(check_board)
    result = tm.predict(check_board)
    return result
    
#tsetlinStandardWeightedHandler(epochs, clauses, T, s)
#tsetlinStandardWeightedHandler(500, 15000, 28000, 15)
# Make sure the settings are equal to those use when training
global tm
tm = tsetlinStandardWeightedHandler(500, 15000, 28000, 15)

"""
board = [1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,3,3,3,3,3,3,3,3,3,3,3,3]
result = evaluate(board)
print(result)
"""


#Returns every legal board the active player can create in the given turn.
def findLegalBoards(board, playerturn):

    
def player1DoTurn(board):


def player2DoTurn(board):

current_board = [1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,3,3,3,3,3,3,3,3,3,3,3,3]

#black player = player 1 = 1
#white player = player 2 = 2
playerturn = 1

#Tells whether the game is over or not. True = the game is going on right now.
active = True


while(active == True):
    
        #Is on the current board
        
        if (playerturn == 1):
            board = player1DoTurn(board)
            
        elif (playerturn == 2):
            #Other player does things.
            board = player2DoTurn(board)
        
