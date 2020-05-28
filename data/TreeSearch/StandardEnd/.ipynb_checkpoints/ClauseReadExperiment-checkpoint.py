from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
import numpy as np
from time import time
import math
from operator import itemgetter

X_train = np.array([])
Y_train = np.array([])
X_test = np.array([])
Y_test = np.array([])
global global_weight_data
global_weight_data = []

#base_path_start = "Data/KfoldDataStaticTransformed/"
base_path_start = "data/"
base_path_end = "statickfoldblack.data"
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
    global global_weight_data
    score = tsetlinStandardWeighted(epochs, clauses, T, s, k_fold_amount)
    print(score)
    mean_score = round(sum(score)/len(score),2)

    #acstr = str(accuracy)
    #with open("StandardTsetlinWeightedLog.txt", "a+") as myfile:
    with open("StandardTsetlinWeightedLogPositiveBoost.txt", "a+") as myfile:
        myfile.write("Clauses: " + str(clauses) + ", Treshold: " + str(T) + ", S: " + str(s) + ", Epochs: " + str(epochs) + ", Mean accuracy: " + str(mean_score) + ", Score per Kfold: " + str(score) + "\n")
    myfile.close()

    print("Clauses: " + str(clauses) + ", Treshold: " + str(T) + ", S: " + str(s) + ", Epochs: " + str(epochs) + ", Mean accuracy: " + str(mean_score) + ", Score per Kfold: " + str(score) + "\n")
    print("Results saved to file")
    tmPrintBestOutputs(global_weight_data, 5)
    


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


def tmStoreOutput(tm, X_test, Y_test, amount_to_print):
    """
    Prosentene for hvert utfall
    Hva riktig utfall er
    """
    global global_weight_data
    test_boards = []
    
    print("Gathering weights")
    
    weights = tm.get_state()
    lossweights = weights[0][0]
    winweights = weights[1][0]
    drawweights = weights[2][0]
    
    for q in range(len(X_test)):
        this_board = []
        for z in range(len(X_test[q])):
            this_board.append(X_test[q][z])
        this_board.append(Y_test[q])
        test_boards.append(this_board)

    tosort = []

    for i in range(len(test_boards)):
        these_clauses = []
        arraytest_boards = np.array([test_boards[i]])
        predictions = tm.transform(arraytest_boards, inverted = False)

        loss = 0
        nloss = 0
        win = 0
        
        nwin = 0
        draw = 0
        ndraw = 0
        
        split = len(predictions[0])/3
        splitchanger = 0
        
        for z in range(len(predictions[0])):
            if(z == split):
                splitchanger += 1
            elif(z == split*2):
                splitchanger += 1
            
            if(splitchanger == 0):
                if(predictions[0][z] == 1):
                    loss += 1 * lossweights[z]
                elif(predictions[0][z] == 0):
                    nloss += 1 * lossweights[z]
                else:
                    print("tmStoreOutput: Clause not loss, win or draw")
                    exit(69)
            elif(splitchanger == 1):
                index = int(z - split)
                if(predictions[0][z] == 1):
                    win += 1 * winweights[index]
                elif(predictions[0][z] == 0):
                    nwin += 1 * winweights[index]
                else:
                    print("tmStoreOutput: Clause not loss, win or draw")
                    exit(69)
            else:
                index = int(z - split*2)
                if(predictions[0][z] == 1):
                    draw += 1 * drawweights[index]
                elif(predictions[0][z] == 0):
                    ndraw += 1 * drawweights[index]
                else:
                    print("tmStoreOutput: Clause not loss, win or draw")
                    exit(69)

        #print("Board number: ", i, " Actual result: ", Y_test[i])
        
        these_clauses.append(i)
        these_clauses.append(Y_test[i])
        #print("Positives: Loss: ", loss, " Win: ", win, " Draw: ", draw)

        loss = round(loss)
        nloss = round(nloss)
        
        win = round(win)
        nwin = round(nwin)
        
        draw = round(draw)
        ndraw = round(ndraw)
        
        these_clauses.append(loss)
        these_clauses.append(win)
        these_clauses.append(draw)
        #print("Negatives: Loss: ", nloss, " Win: ", nwin, " Draw: ", ndraw)
        
        these_clauses.append(nloss)
        these_clauses.append(nwin)
        these_clauses.append(ndraw)
        loss_score = loss - nloss
        win_score = win - nwin
        draw_score = draw - ndraw
        #print("Sum of predictions: Loss: ", loss_score, " Win: ", win_score, " Draw: ", draw_score)
        
        these_clauses.append(loss_score)
        these_clauses.append(win_score)
        these_clauses.append(draw_score)
        outcome = 8
        weightscore = 8
        difference_sum = 0

        if(loss_score > win_score and loss_score > draw_score):
            outcome = 0
            weightscore = loss_score
            difference_sum = (loss_score - win_score) + (loss_score - draw_score)
        elif(win_score > loss_score and win_score > draw_score):
            outcome = 1
            weightscore = win_score
            difference_sum = (win_score - loss_score) + (win_score - draw_score)
        elif(draw_score > loss_score and draw_score > win_score):
            outcome = 2
            weightscore = draw_score
            difference_sum = (draw_score - loss_score) + (draw_score - win_score)
        else:
            if(loss_score == win_score and loss_score == draw_score):
                print("tmStoreOutput, all weight scores are the same")
                #weightscore coulb be any as they are all equal, chose loss as it comes first.
                outcome = "69"
                weightscore = loss_score
                difference_sum = 0
                
            elif(loss_score == win_score and loss_score != draw_score):
                outcome = "01"
                #weightscore coulb be any as they are all equal, chose loss as it comes first.
                weightscore = loss_score
                difference_sum = loss_score - draw_score
            
            elif(loss_score == draw_score and loss_score != win_score):
                outcome = "02"
                #weightscore coulb be any as they are all equal, chose loss as it comes first.
                weightscore = loss_score
                difference_sum = loss_score - win_score
            
            elif(win_score == draw_score and win_score != loss_score):
                outcome = "12"
                #weightscore coulb be any as they are all equal, chose loss as it comes first.
                weightscore = win_score
                difference_sum = win_score - loss_score
            
            else:
                print("Error in tmStoreOutput, differences in scores; they are not all equal, yet not just two of them are equal? Debugs:")
                print("Loss score: ", loss_score, " Win score: ", win_score, " Draw score: ", draw_score)
                exit(234)
            
                
        #print("Tsetlin thinks ", outcome, " is correct, with score ", weightscore)
        these_clauses.append(outcome)
        these_clauses.append(weightscore)
        these_clauses.append(difference_sum)

        tosort.append(these_clauses)

    print("Sorting scores, total scores to sort: ", len(tosort))
    tostore = []

    sorted_list = sorted(tosort, reverse=True, key=itemgetter(13))
    print("List sorted, total sorted ", len(sorted_list))

    for i in range(amount_to_print):
        tostore.append(sorted_list[i])

    for k in range(len(tostore)):
        global_weight_data.append(tostore[k])
    return

def tmPrintBestOutputs(outputs, amount_to_print):

    print("Started tmPrintBestOutputs, printing ", amount_to_print, " best scores for boards")
    print("Sorting scores, total scores to sort: ", len(outputs))
    toprint = []
    
    sorted_list = sorted(outputs, reverse=True, key=itemgetter(13)   )
    print("List sorted, total sorted ", len(sorted_list))
    
    for i in range(amount_to_print):
        toprint.append(sorted_list[i])
        
    toprint.append(sorted_list[-1])
    for x in range(len(toprint)):
        print("Printing clauses for best board number ", x)
        print("DEBUG: Length of this list is ", len(toprint[x]))
        print("Board number: ", toprint[x][0], " Actual result: ", toprint[x][1])
        print("Positives: Loss: ", toprint[x][2], " Win: ", toprint[x][3], " Draw: ", toprint[x][4])
        print("Negatives: Loss: ", toprint[x][5], " Win: ", toprint[x][6], " Draw: ", toprint[x][7])
        print("Sum of predictions: Loss: ", toprint[x][8], " Win: ", toprint[x][9], " Draw: ", toprint[x][10])
        print("Tsetlin thinks ", toprint[x][11], " is correct, with score ", toprint[x][12], " giving a difference value of ", toprint[x][13])
    
    return
                             
    
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
    tmStoreOutput(tm, X_test, Y_test, 5)
    
    print("Finished running.. \n")


    return mean_accuracy



# Parameters
# split_ratio = 0.9

k_fold_amount = 10
#tsetlinStandardWeightedHandler(epochs, clauses, T, s, k_fold_amount)


#tsetlinStandardWeightedHandler(500, 15000, 40000, 15, k_fold_amount)
tsetlinStandardWeightedHandler(500, 10, 5000, 15, k_fold_amount)
