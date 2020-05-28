from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
import numpy as np
from time import time
import math
from operator import itemgetter

X_train = np.array([])
Y_train = np.array([])
X_test = np.array([])
Y_test = np.array([])

# base_path_start = "Data/KfoldDataStaticTransformed/"
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
    result = round(100 * (tm.predict(X_test) == Y_test).mean(), 2)
    print("Trained accuracy: ", result)
    return tm



global lossweightsblack
global winweightsblack
global drawweightsblack

global lossweightswhite
global winweightswhite
global drawweightswhite


def TMInitBlack(epochs, clauses, treshold, s):
    print("Initializing Black Tsetlin")
    global base_path_end
    base_path_end = "statickfoldblack.data"
    global statefile
    statefile = "SavedStatesBlack.npy"
    global tmblack
    tmblack = tsetlinStandardWeightedHandler(epochs, clauses, treshold, s)
    global lossweightsblack
    global winweightsblack
    global drawweightsblack
    weightsblack = tmblack.get_state()
    lossweightsblack = weightsblack[0][0]
    winweightsblack = weightsblack[1][0]
    drawweightsblack = weightsblack[2][0]



def TMInitWhite(epochs, clauses, treshold, s):
    print("Initializing White Tsetlin")
    global base_path_end
    base_path_end = "statickfoldwhite.data"
    global statefile
    statefile = "SavedStatesWhite.npy"
    global tmwhite
    tmwhite = tsetlinStandardWeightedHandler(epochs, clauses, treshold, s)
    global lossweightswhite
    global winweightswhite
    global drawweightswhite
    weightswhite = tmwhite.get_state()
    lossweightswhite = weightswhite[0][0]
    winweightswhite = weightswhite[1][0]
    drawweightswhite = weightswhite[2][0]



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



def getResultScore(board, result, tm, color):
    weight_data = []
    test_boards = []

    lossweights = []
    winweights = []
    drawweights = []

    if (color == "black"):
        global lossweightsblack
        global winweightsblack
        global drawweightsblack
        lossweights = lossweightsblack
        winweights = winweightsblack
        drawweights = drawweightsblack
    elif (color == "white"):
        global lossweightswhite
        global winweightswhite
        global drawweightswhite
        lossweights = lossweightswhite
        winweights = winweightswhite
        drawweights = drawweightswhite
    else:
        print("getResultScore wrong color parameter, should be black or white")
        exit(34)

    loc_board = board.copy()
    loc_board.append(result)

    #these_clauses = []
    arraytest_boards = np.array([loc_board])
    predictions = tm.transform(arraytest_boards, inverted=False)

    loss = 0
    nloss = 0
    win = 0

    nwin = 0
    draw = 0
    ndraw = 0

    split = len(predictions[0]) / 3
    splitchanger = 0

    for z in range(len(predictions[0])):
        if (z == split):
            splitchanger += 1
        elif (z == split * 2):
            splitchanger += 1

        if (splitchanger == 0):
            if (predictions[0][z] == 1):
                loss += 1 * lossweights[z]
            elif (predictions[0][z] == 0):
                nloss += 1 * lossweights[z]
            else:
                print("tmStoreOutput: Clause not loss, win or draw")
                exit(69)
        elif (splitchanger == 1):
            index = int(z - split)
            if (predictions[0][z] == 1):
                win += 1 * winweights[index]
            elif (predictions[0][z] == 0):
                nwin += 1 * winweights[index]
            else:
                print("tmStoreOutput: Clause not loss, win or draw")
                exit(69)
        else:
            index = int(z - split * 2)
            if (predictions[0][z] == 1):
                draw += 1 * drawweights[index]
            elif (predictions[0][z] == 0):
                ndraw += 1 * drawweights[index]
            else:
                print("tmStoreOutput: Clause not loss, win or draw")
                exit(69)

    # print("Board number: ", i, " Actual result: ", Y_test[i])

    #these_clauses.append(result)
    # print("Positives: Loss: ", loss, " Win: ", win, " Draw: ", draw)

    loss = round(loss)
    nloss = round(nloss)

    win = round(win)
    nwin = round(nwin)

    draw = round(draw)
    ndraw = round(ndraw)

    #these_clauses.append(loss)
    #these_clauses.append(win)
    #these_clauses.append(draw)
    # print("Negatives: Loss: ", nloss, " Win: ", nwin, " Draw: ", ndraw)

    #these_clauses.append(nloss)
    #these_clauses.append(nwin)
    #these_clauses.append(ndraw)
    loss_score = loss - nloss
    win_score = win - nwin
    draw_score = draw - ndraw
    # print("Sum of predictions: Loss: ", loss_score, " Win: ", win_score, " Draw: ", draw_score)

    #these_clauses.append(loss_score)
    #these_clauses.append(win_score)
    #these_clauses.append(draw_score)
    #outcome = 8
    #weightscore = 8
    difference_sum = 0

    if (loss_score > win_score and loss_score > draw_score):
        #outcome = 0
        #weightscore = loss_score
        difference_sum = (loss_score - win_score) + (loss_score - draw_score)
    elif (win_score > loss_score and win_score > draw_score):
        #outcome = 1
        #weightscore = win_score
        difference_sum = (win_score - loss_score) + (win_score - draw_score)
    elif (draw_score > loss_score and draw_score > win_score):
        #outcome = 2
        #weightscore = draw_score
        difference_sum = (draw_score - loss_score) + (draw_score - win_score)
    else:
        if (loss_score == win_score and loss_score == draw_score):
            print("tmStoreOutput, all weight scores are the same")
            # weightscore could be any as they are all equal, chose loss as it comes first.
            #outcome = "69"
            #weightscore = loss_score
            difference_sum = 0

        elif (loss_score == win_score and loss_score != draw_score):
            #outcome = "01"
            # weightscore could be any as they are all equal, chose loss as it comes first.
            #weightscore = loss_score
            difference_sum = loss_score - draw_score

        elif (loss_score == draw_score and loss_score != win_score):
            #outcome = "02"
            # weightscore could be any as they are all equal, chose loss as it comes first.
            #weightscore = loss_score
            difference_sum = loss_score - win_score

        elif (win_score == draw_score and win_score != loss_score):
            #outcome = "12"
            # weightscore could be any as they are all equal, chose loss as it comes first.
            #weightscore = win_score
            difference_sum = win_score - loss_score

        else:
            print(
                "Error in tmStoreOutput, differences in scores; they are not all equal, yet not just two of them are equal? Debugs:")
            print("Loss score: ", loss_score, " Win score: ", win_score, " Draw score: ", draw_score)
            exit(234)

    # print("Tsetlin thinks ", outcome, " is correct, with score ", weightscore)
    #these_clauses.append(outcome)
    #these_clauses.append(weightscore)
    #these_clauses.append(difference_sum)

    return round(difference_sum/1000, 0)

                                 

    
#Evaluates the board, returns loss 0, win 1 or draw 0.5
def multiEvaluateWeighted(boards, tm, color):
    #tm is either tmblack or tmwhite. tmblack evaluates the winning chance of black player. tmwhite evaluates the winning chance of white player.
    check_boards = np.asarray(boards)
    result = tm.predict(check_boards)
    #Draw returns 0.5, a value or score between loss and win.
    results = []
    #The scores are modiefied to make sure that the scores of wins are always higher than scores draws, and scores of draws always higher than those of losses.
    for i in range(len(result)):
        # Draw
        if (result[i]  == 2):
            this_score = getResultScore(boards[i], result[i], tm, color)
            this_score += 2000
            this_score = round(this_score/100)
            results.append(this_score)
        #Win
        elif (result[i] == 1):
            this_score = getResultScore(boards[i], result[i], tm, color)
            this_score += 5000
            this_score = round(this_score/100)
            results.append(this_score)                     
        #Loss
        elif (result[i] == 0):
            #Negated by multiplying with (-1) as thinking that the loss prediction is reliable means that it might be more likely to be a good prediction. If it is a good prediction, we really do not want to get in this situation.
            this_score = getResultScore(boards[i], result[i], tm, color) * (-1)
            this_score -= 1000
            this_score = round(this_score/100)
            results.append(this_score)
        else:
            print("Result in evaluate not 2, 1 nor 0. This should not happen")
            exit(69)
    return results









# tsetlinStandardWeightedHandler(epochs, clauses, T, s)
# tsetlinStandardWeightedHandler(500, 15000, 28000, 15)

# Make sure the settings are equal to those used when training
#TMInitBlack(500, 15000, 28000, 15)
TMInitBlack(500, 15000, 28000, 15)
print(" ")
print(" ")
#TMInitWhite(500, 15000, 28000, 15)

# Clear the Tsetlin initializing data to relieve the memory a little.
X_train = 0
Y_train = 0
X_test = 0
Y_test = 0

print(" ")


current_board = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]

#print(multiEvaluate([current_board], tmblack))

global global_weight_data
global_weight_data = []



#weights = tmblack.get_state()

#lossweights = weights[0][0]
#winweights = weights[1][0]
#drawweights = weights[2][0]

#print(weights)



boards = []
boards.append(current_board)
boards.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
boards.append([1,2,1,0,0,0,0,0,1,0,1,0,2,0,2,0,2,0,0,0,0,3,0,2,0,0,0,3,0,0,0,0])
boards.append([0,0,0,0,0,0,0,0,3,0,0,4,0,0,0,4,0,0,0,0,0,0,3,0,0,0,0,0,0,0,0,0])
boards.append([0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,0,0])
boards.append([1,1,1,1,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,3,3,3])
boards.append([4,4,1,4,0,0,0,0,0,0,0,0,0,0,0,2,1,0,0,1,0,0,0,0,0,0,0,0,0,0,3,0])
boards.append([0,0,4,0,0,0,0,0,0,0,0,0,0,3,0,0,0,3,3,3,4,0,0,3,0,3,0,3,3,0,0,0])
boards.append([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
boards.append([3,3,3,3,3,3,3,3,3,3,3,3,3,3,4,3,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3])


"""
test_this = np.asarray([[1,0,0,0,0,0,0,0,0,0,1,1,3,3,0,0,0,0,0,0,0,0,3,0,0,0,0,0,0,0,0,0]])
result = tmblack.predict(test_this)
print(result)
"""


print(multiEvaluateWeighted(boards, tmblack, "black"))
exit(22)    
    



