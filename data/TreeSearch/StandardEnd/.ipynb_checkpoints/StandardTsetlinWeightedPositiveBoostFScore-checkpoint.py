from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
import numpy as np
from time import time
import math

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

    #This MUST be the same as when training str(number)
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
    TM = tsetlinStandardWeighted(epochs, clauses, T, s)
    print("TM successfully prepared")
    return TM





def tsetlinStandardWeightedHandler(epochs, clauses, T, s):
    TM = tsetlinStandardWeighted(epochs, clauses, T, s)
    #print(score)
    #mean_score = round(sum(score)/len(score),2)
    score = 22
    mean_score = 23
    
    
    print(" ")
    print("Predicted wins:")
    print(predicted_wins)
    print("Correctly predicted wins")
    print(correctly_predicted_wins)
    print("Total wins")
    print(total_wins)

    win_precisions = []
    win_recalls = []
    win_FScores = []

    for i in range(len(total_wins)):
        precision = correctly_predicted_wins[i]/predicted_wins[i]
        recall = correctly_predicted_wins[i]/total_wins[i]
        FScore = 2*((precision*recall)/(precision+recall))
        print("Win FScore for K-fold number ", i, " is ", round(FScore, 2))
        win_precisions.append(round(precision, 2))
        win_recalls.append(round(recall, 2))
        win_FScores.append(round(FScore, 2))
    print("Win precisions avg ", round((sum(win_precisions)/len(win_precisions)), 2))
    print(win_precisions)
    print("Win recalls avg ", round((sum(win_recalls)/len(win_recalls)), 2))
    print(win_recalls)
    print("Win FScores avg ", round((sum(win_FScores)/len(win_FScores)), 2))
    print(win_FScores)
    print(" ")


    print(" ")
    print("Predicted losses:")
    print(predicted_losses)
    print("Correctly predicted losses")
    print(correctly_predicted_losses)
    print("Total losses")
    print(total_losses)

    loss_precisions = []
    loss_recalls = []
    loss_FScores = []

    for i in range(len(total_losses)):
        precision = correctly_predicted_losses[i]/predicted_losses[i]
        recall = correctly_predicted_losses[i]/total_losses[i]
        FScore = 2*((precision*recall)/(precision+recall))
        print("Loss FScore for K-fold number ", i, " is ", round(FScore, 2))
        loss_precisions.append(round(precision, 2))
        loss_recalls.append(round(recall, 2))
        loss_FScores.append(round(FScore, 2))
    print("Loss precisions avg ", round((sum(loss_precisions)/len(loss_precisions)), 2))
    print(loss_precisions)
    print("Loss recalls avg ", round((sum(loss_recalls)/len(loss_recalls)), 2))
    print(loss_recalls)
    print("Loss FScores avg ", round((sum(loss_FScores)/len(loss_FScores)), 2))
    print(loss_FScores)
    print(" ")


    print(" ")
    print("Predicted draws:")
    print(predicted_draws)
    print("Correctly predicted draws")
    print(correctly_predicted_draws)
    print("Total draws")
    print(total_draws)

    draw_precisions = []
    draw_recalls = []
    draw_FScores = []

    for i in range(len(total_draws)):
        precision = correctly_predicted_draws[i]/predicted_draws[i]
        recall = correctly_predicted_draws[i]/total_draws[i]
        FScore = 2*((precision*recall)/(precision+recall))
        print("Draw FScore for K-fold number ", i, " is ", round(FScore, 2))
        draw_precisions.append(round(precision, 2))
        draw_recalls.append(round(recall, 2))
        draw_FScores.append(round(FScore, 2))
    print("Draw precisions avg ", round((sum(draw_precisions)/len(draw_precisions)), 2))
    print(draw_precisions)
    print("Draw recalls avg ", round((sum(draw_recalls)/len(draw_recalls)), 2))
    print(draw_recalls)
    print("Draw FScores avg ", round((sum(draw_FScores)/len(draw_FScores)), 2))
    print(draw_FScores)
    print(" ")
    
    
    

    
    #acstr = str(accuracy)
    #with open("StandardTsetlinWeightedLog.txt", "a+") as myfile:
    with open("STATSStandardTsetlinWeightedLogPositiveBoost.txt", "a+") as myfile:
        myfile.write("\n")
        myfile.write("Dataset: StandardEnd" + ", Clauses: " + str(clauses) + ", Treshold: " + str(T) + ", S: " + str(s) + ", Epochs: " + str(epochs) + ", Mean accuracy: " + str(mean_score) + ", Score per Kfold: " + str(score) + "\n")
        myfile.write("Win precision avg and list: " + str(round((sum(win_precisions)/len(win_precisions)), 2)) + " " + str(win_precisions) + "\n")
        myfile.write("Win recalls avg and list: " + str(round((sum(win_recalls)/len(win_recalls)), 2)) + " " + str(win_recalls) + "\n")
        myfile.write("Win FScores avg and list: " + str(round((sum(win_FScores)/len(win_FScores)), 2)) + " " + str(win_FScores) + "\n")
        myfile.write("Loss precision avg and list: " + str(round((sum(loss_precisions)/len(loss_precisions)), 2)) + " " + str(loss_precisions) + "\n")
        myfile.write("Loss recalls avg and list: " + str(round((sum(loss_recalls)/len(loss_recalls)), 2)) + " " + str(loss_recalls) + "\n")
        myfile.write("Loss FScores avg and list: " + str(round((sum(loss_FScores)/len(loss_FScores)), 2)) + " " + str(loss_FScores) + "\n")
        myfile.write("Draw precision avg and list: " + str(round((sum(draw_precisions)/len(draw_precisions)), 2)) + " " + str(draw_precisions) + "\n")
        myfile.write("Draw recalls avg and list: " + str(round((sum(draw_recalls)/len(draw_recalls)), 2)) + " " + str(draw_recalls) + "\n")
        myfile.write("Draw FScores avg and list: " + str(round((sum(draw_FScores)/len(draw_FScores)), 2)) + " " + str(draw_FScores) + "\n")
        myfile.write("\n")
    myfile.close()
    
    
    

    
    print("Clauses: " + str(clauses) + ", Treshold: " + str(T) + ", S: " + str(s) + ", Epochs: " + str(epochs) + ", Mean accuracy: " + str(mean_score) + ", Score per Kfold: " + str(score) + "\n")
    print("Results saved to file")


    


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
    
    multiScoreFind(X_test, Y_test, tm)
    
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
    
    
    
    
    
# Evaluates the board
def multiScoreFind(boards, answers, tm):
    print("Evaluating....")
    # check_boards = np.asarray(boards)
    # result = tm.predict(check_boards)
    result = tm.predict(boards)

    global predicted_losses
    global correctly_predicted_losses
    global total_losses

    this_predicted_losses = 0
    this_correctly_predicted_losses = 0
    this_total_losses = 0

    global predicted_wins
    global correctly_predicted_wins
    global total_wins

    this_predicted_wins = 0
    this_correctly_predicted_wins = 0
    this_total_wins = 0

    global predicted_draws
    global correctly_predicted_draws
    global total_draws

    this_predicted_draws = 0
    this_correctly_predicted_draws = 0
    this_total_draws = 0

    for i in range(len(result)):
        # Draw
        if (result[i] == 2):
            this_predicted_draws += 1
            if (answers[i] == 2):
                # correctly predicted
                this_correctly_predicted_draws += 1
        # Win
        elif (result[i] == 1):
            this_predicted_wins += 1
            if (answers[i] == 1):
                # correctly predicted
                this_correctly_predicted_wins += 1
        # Loss
        elif (result[i] == 0):
            this_predicted_losses += 1
            if (answers[i] == 0):
                # correctly predicted
                this_correctly_predicted_losses += 1
        else:
            print("Result in evaluate not 2, 1 nor 0. This should not happen")
            exit(69)

    for i in range(len(answers)):
        if (answers[i] == 2):
            this_total_draws += 1
        elif (answers[i] == 1):
            this_total_wins += 1
        elif (answers[i] == 0):
            this_total_losses += 1
        else:
            print("Answer not 2, 1 nor 0. This should not happen")
            exit(69)

    predicted_losses.append(this_predicted_losses)
    correctly_predicted_losses.append(this_correctly_predicted_losses)
    total_losses.append(this_total_losses)

    predicted_wins.append(this_predicted_wins)
    correctly_predicted_wins.append(this_correctly_predicted_wins)
    total_wins.append(this_total_wins)

    predicted_draws.append(this_predicted_draws)
    correctly_predicted_draws.append(this_correctly_predicted_draws)
    total_draws.append(this_total_draws)
    




global predicted_losses
global correctly_predicted_losses
global total_losses
predicted_losses = []
correctly_predicted_losses = []
total_losses = []

global predicted_wins
global correctly_predicted_wins
global total_wins
predicted_wins = []
correctly_predicted_wins = []
total_wins = []

global predicted_draws
global correctly_predicted_draws
global total_draws
predicted_draws = []
correctly_predicted_draws = []
total_draws = []

    
    
    
    

# tsetlinStandardWeightedHandler(epochs, clauses, T, s)
# tsetlinStandardWeightedHandler(500, 15000, 28000, 15)

# Make sure the settings are equal to those used when training
TMInitBlack(500, 19000, 40000, 9)
print(" ")
print(" ")
TMInitWhite(500, 19000, 40000, 9)

# Clear the Tsetlin initializing data to relieve the memory a little.
X_train = 0
Y_train = 0
X_test = 0
Y_test = 0












