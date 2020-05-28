import csv
import math
import numpy as np

# my_data = np.genfromtxt('OCA_2.0.csv', delimiter=',')
"""Create the starting board and define what data represents what type of piece"""
print("Extracting moves")
# print(my_data)
# [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# 0 = empty
# 1 = black checker
# 2 = black king
# 3 = white checker
# 4 = white king
# starting_board = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]

# print(len(starting_board))
# print(starting_board)


"""Clean the source data"""
#       0-1 = 0 = white wins
#   1-0 = 1 = black wins
#   1/2-1/2 = 2 = draw


wins = 0
losses = 0
draws = 0

bad_boards = 0

"""
playBuilder function:
Inputs a list of lines containing plays and the match result
Outputs a list of two items. 
0: list of all moves, one move per index.
1: match result.
"""


def playBuilderEngine(local_playstring):
    somelist = []
    capturetimeout = 0

    for q in range(len(local_playstring)):
        thisaction = ""
        if (local_playstring[q] == "-"):
            if (q > 0):
                if (local_playstring[q - 1].isdigit()):
                    if (q > 1):
                        if (local_playstring[q - 2].isdigit()):
                            thisaction = thisaction + local_playstring[q - 2]
                    thisaction = thisaction + local_playstring[q - 1]
                    thisaction = thisaction + local_playstring[q]
            if (q < len(local_playstring) - 1):
                if (local_playstring[q + 1].isdigit()):
                    thisaction = thisaction + local_playstring[q + 1]
                    if (q < len(local_playstring) - 2):
                        if (local_playstring[q + 2].isdigit()):
                            thisaction = thisaction + local_playstring[q + 2]
                    somelist.append(thisaction)
        elif (local_playstring[q] == "x"):
            if (capturetimeout == 0):
                capturetimeout = 1
                if (q > 0):
                    if (local_playstring[q - 1].isdigit()):
                        if (q > 1):
                            if (local_playstring[q - 2].isdigit()):
                                thisaction = thisaction + local_playstring[q - 2]
                        thisaction = thisaction + local_playstring[q - 1]
                        thisaction = thisaction + local_playstring[q]
                w = 1
                while (w < len(local_playstring)):
                    iter = q + w
                    w += 1
                    if (iter < len(local_playstring)):
                        if (local_playstring[iter].isdigit()):
                            thisaction = thisaction + local_playstring[iter]
                        elif (local_playstring[iter] == "x"):
                            if (local_playstring[iter - 1] == "x"):
                                print("ERROR 1, double x at index: ", iter - 1)
                                exit(1)
                            else:
                                thisaction = thisaction + local_playstring[iter]
                        else:
                            w = len(local_playstring) + 20
                    else:
                        w = len(local_playstring) + 20
                somelist.append(thisaction)
        elif (local_playstring[q] == " "):
            capturetimeout = 0
    return somelist


def playBuilder(local_playlist, local_result):
    product_list = []
    listed_moves = []
    for i in range(len(local_playlist)):
        list_plays = playBuilderEngine(local_playlist[i])
        for z in range(len(list_plays)):
            listed_moves.append(list_plays[z])
    product_list.append(listed_moves)
    product_list.append(local_result)
    return product_list


# lines = open("minidata.csv").read().splitlines()
lines = open("OCA_2.0.csv").read().splitlines()
# print(f.readlines())
datapoints = 0
fetch_moves = 0
extracted_data = []
this_result = 0
this_play = []

loaded_datapoints = 0
bad_data = 0

for i in range(len(lines)):
    # print(lines[i])
    if (lines[i].__contains__("Result")):
        if (lines[i].__contains__("0-1")):
            this_result = 0
            losses += 1
        elif (lines[i].__contains__("1-0")):
            this_result = 1
            wins += 1
        elif (lines[i].__contains__("1/2-1/2")):
            this_result = 2
            draws += 1
        else:
            print("ERROR 2 on number ", i, "undefined Result: ", lines[i])
            exit(2)
        fetch_moves = 1
        datapoints += 1
        # print("Result found: ", this_result)
    elif (fetch_moves != 0):
        # if(not(lines[i][0] == )):
        # print("Fetching moves")
        if (len(lines[i]) > 0):
            if (lines[i][0].isdigit()):
                # print("Adding move")
                this_play.append(lines[i])
        else:
            # print("Finalizing play")
            if this_play[-1].__contains__("0-1"):
                this_play[-1] = this_play[-1][:- 4]
            elif this_play[-1].__contains__("1-0"):
                this_play[-1] = this_play[-1][:- 4]
            elif this_play[-1].__contains__("1/2-1/2"):
                this_play[-1] = this_play[-1][:- 8]
            else:
                print("ERROR 3, no extra result to remove at i number: ", i)
                print(this_play[-1])
                print("Data point ignored")
                bad_data = 1
                # exit(3)

            if (bad_data == 1):
                bad_boards += 1
                this_play = []
                fetch_moves = 0
                bad_data = 0
            else:
                this_play = playBuilder(this_play, this_result)
                extracted_data.append(this_play)
                this_play = []
                fetch_moves = 0
                loaded_datapoints += 1

this_play.append(this_result)
extracted_data.append(this_play)
loaded_datapoints += 1

# print("Extracted data: ")
# for i in range(len(extracted_data)):
# print(extracted_data[i])

# print("Datapoints: ", datapoints)
# print("Loaded datapoints: ", loaded_datapoints)

# One full datapoint
print(extracted_data[0])
# One datapoint's list of moves
print(extracted_data[0][0])
# One datapoint's result
print(extracted_data[0][1])

print("Wins: ", wins)
print("Losses: ", losses)
print("Draws: ", draws)

print(len(extracted_data))

"""
#Save to file:
import csv

writer = csv.writer(open("cleandata.csv", 'w'))
for row in extracted_data:
    writer.writerow(row)
"""

# Makes KFold

print(extracted_data[0][0])
print(extracted_data[0][1])


"""
def removeDuplicates(raw_list):
    print("Checking for duplicates")
    no_duplicate_list = []
    loss = 0
    win = 0
    draw = 0
    original_length = len(raw_list)

    for q in range(len(raw_list)):
        unique = 1
        if(len(raw_list[q]) == 2):
            for w in range(len(raw_list)):
                if(q != w and raw_list[q] == raw_list[w]):
                    unique = 0
            if(unique == 1):
                no_duplicate_list.append(raw_list[q])
                if(raw_list[q][1] == 0):
                    loss += 1
                elif (raw_list[q][1] == 1):
                    win += 1
                elif (raw_list[q][1] == 2):
                    draw += 1
                else:
                    print("ERROR 15, No result to record")
                    exit(15)
            else:
                if(not no_duplicate_list.__contains__(raw_list[q])):
                    no_duplicate_list.append(raw_list[q])
                    if (raw_list[q][-1] == 0):
                        loss += 1
                    elif (raw_list[q][-1] == 1):
                        win += 1
                    elif (raw_list[q][-1] == 2):
                        draw += 1
                    else:
                        print("ERROR 15, No result to record")
                        exit(15)
    print("Losses: ", loss)
    print("Winds: ", win)
    print("Draws: ", draw)
    print("Duplicates removed: ", original_length - len(no_duplicate_list))
    print("Total data: ", len(no_duplicate_list))
    return no_duplicate_list



print("yadd")
print(extracted_data[0])
print(len(extracted_data))
extracted_data = removeDuplicates(extracted_data)
print("yodd")
print(extracted_data[0])
print(len(extracted_data))


"""


"""

Extracte moves, ok
duplicate check, ok
kfold, must do
binary, must do

Problem? Kan ikke sjekke for alle mulige duplikater etter omgjøring til brett da fordelingen allerede er gjort. Det hadde gått om jeg ikke skulle hatt ganske like kfold størrelser og at brettene dannet på samme datapunkt kunne vært i både test og train.

Duplikat-sjekk: Så lenge det kun finnes én versjon i hver kfold, altså at det samme brettet ikke finnes i både train og test, så er det ok mtp. å holde test ren. Derfor tar jeg bare en sjekk på både train og test respektivt kanskje? Må kanskje refordele enda en gang for å få en 90/10 split.

Løsning: det er ikke et problem. eneste problemet her er at duplikat-sjekken må kjøres for hvert sett av train og test. Dette tar en del tid å kjøre gjennom. I tillegg så kan splitten bli litt rar, ikke helt 90%

"""


def round_down(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier) / multiplier


"""Build the boards, while checking for illegal moves as a way to counteract possible typing errors"""


# There are multiple ways to create data of this.
# Must do
# Create final board with the result
# Maybe do
# Create multiple boards with some result. Possibly average out equal boards in case equal boards may lead to
# different results or simply allow "duplicates" with different results.
# The power of Tsetlin machine may find out that three equal boards with
# different outcomes states that the outcome is uncertain?
# Granmo tips regarding data making:
# You can create more data by using previous moves, BUT: make sure to keep the bonus data
# in the same data category (training/test) in order to not contaminate the test data.

# In theory:
# [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
# In programming:
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# 0 = empty
# 1 = black checker
# 2 = black king
# 3 = white checker
# 4 = white king
# starting_board = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]


#    1   2   3   4
#  5   6   7   8
#    9   10  11  12
#  13  14  15  16
#    17  18  19  20
#  21  22  23  24
#    25  26  27  28
#  29  30  31  32

def getDiagonalTable():
    diagonals = [[5, 6], [6, 7], [7, 8], [8], [1, 9], [1, 2, 9, 10], [2, 3, 10, 11],
                 [3, 4, 11, 12], [5, 6, 13, 14], [6, 7, 14, 15], [7, 8, 15, 16], [8, 16],
                 [9, 17], [9, 10, 17, 18], [10, 11, 18, 19], [11, 12, 19, 20],
                 [13, 14, 21, 22], [14, 15, 22, 23], [15, 16, 23, 24], [16, 24],
                 [17, 25], [17, 18, 25, 26], [18, 19, 26, 27], [19, 20, 27, 28],
                 [21, 22, 29, 30], [22, 23, 30, 31], [23, 24, 31, 32], [24, 32],
                 [25], [25, 26], [26, 27], [27, 28]]
    return diagonals


def isDiagonal(position1, position2):
    local_diagonals = getDiagonalTable()
    if (position2 in local_diagonals[position1 - 1] and position1 != 0 and position2 != 0):
        return True
    else:
        return False


def isMovingRightWay(checker_to_move, moveto, piece_type, board):
    if (piece_type not in [1, 2, 3, 4]):
        print("Error 10, trying to move a non-existing piece, piece is: ", piece_type, " location: ", checker_to_move)
        return False
    if (piece_type == 1):
        if (checker_to_move > moveto):
            print("Wrong way")
            return False
    elif (piece_type == 3):
        if (checker_to_move < moveto):
            print("Wrong way")
            return False
    return True


def movePiece(board, action, checker_to_move):
    moveto = str(action[action.index("-") + 1])
    if (action.index("-") + 2 < len(action)):
        moveto = moveto + str(action[action.index("-") + 2])
    moveto = int(moveto)
    if (board[moveto - 1] == 0):
        if (isDiagonal(checker_to_move, moveto) and isMovingRightWay(checker_to_move, moveto,
                                                                     board[checker_to_move - 1], board)):
            board[moveto - 1] = board[checker_to_move - 1]
            board[checker_to_move - 1] = 0
            upper_border = [1, 2, 3, 4]
            lower_border = [29, 30, 31, 32]
            if (upper_border.__contains__(moveto) and board[moveto - 1] == 3):
                board[moveto - 1] = 4
            elif (lower_border.__contains__(moveto) and board[moveto - 1] == 1):
                board[moveto - 1] = 2
            return board, 1
        else:
            print("Error 9, trying to move piece to non-diagonal spot or wrong way, checker_to_move: ", checker_to_move,
                  " moveto: ", moveto, " board[checker_to_move-1]: ", board[checker_to_move - 1])
            return board, 0
    else:
        print("Error 8, trying to move piece to an occupied spot")
        return board, 0


def isCaptureDiagonal(position1, position2):
    position1 += 1
    position2 += 1
    if ((position1 - position2) == 9 or (position2 - position1) == 9 or (position1 - position2) == 7 or (
            position2 - position1) == 7):
        return True
    else:
        return False


"""This function actually performs the capture action
    Returns board"""


def doCapture(board, position1, position2):
    # Checks if captures opposite color
    diagonals = getDiagonalTable()
    if ((len(list(set(diagonals[position1 - 1]).intersection(diagonals[position2 - 1])))) == 1):
        to_capture = list(set(diagonals[position1 - 1]).intersection(diagonals[position2 - 1]))[0]
    else:
        print("Error 12, found none or multiple pieces to capture when comparing diagonals")
        return board, 0
    black_pieces = [1, 2]
    white_pieces = [3, 4]
    if (board[to_capture - 1] == 0):
        print("Error 14, trying to capture blank position")
        return board, 0
    elif ((board[position1 - 1] in black_pieces and board[to_capture - 1] in black_pieces) or (
            board[position1 - 1] in white_pieces and board[to_capture - 1] in white_pieces)):
        print("Error 13, trying to capture piece of same color")
        return board, 0
    else:
        board[position2 - 1] = board[position1 - 1]
        board[position1 - 1] = 0
        board[to_capture - 1] = 0
        upper_border = [1, 2, 3, 4]
        lower_border = [29, 30, 31, 32]
        if (upper_border.__contains__(position2) and board[position2 - 1] == 3):
            board[position2 - 1] = 4
        elif (lower_border.__contains__(position2) and board[position2 - 1] == 1):
            board[position2 - 1] = 2
    return board, 1


"""This function is responsible for checking for the legality of a single capture
    as well as actually performing the changes to the board.
    Returns boolean"""


def isLegalCapture(board, position1, position2):
    # Make sure that the jump is ok:
    # Does jump over enemy piece
    # Is diagonal, with extra step
    if (isMovingRightWay(position1, position2, board[position1 - 1], board) and isCaptureDiagonal(position1,
                                                                                                  position2)):
        return True
    else:
        return False


"""This function is responsible for handling the captures of pieces.
    It will dissect all capture actions into single capture actions
    to send to isLegalCapture"""


def capturePiece(board, action, checker_to_move):
    # Create a custom function, isLegalCapture.
    position1 = checker_to_move
    first_position_switch = 0
    errorcheck = 1
    for i in range(len(action)):
        if (action[i].__contains__("x")):
            if ((i + 1) < len(action)):
                if (action[i + 1].isdigit()):
                    if (first_position_switch == 1):
                        position1 = position2
                    position2 = str(action[i + 1])
                    if ((i + 2) < len(action)):
                        if (action[i + 2].isdigit()):
                            position2 = position2 + str(action[i + 2])
                    position2 = int(position2)
                    if (isLegalCapture(board, position1, position2)):
                        board, errorcheck = doCapture(board, position1, position2)
                        if (errorcheck == 0):
                            return board, 0
                        first_position_switch = 1
                    else:
                        print(
                            "Error 11, illegal capture attempted, wrong direction or not proper capture diagonal, action: ",
                            action, " position1: ", position1, " position2: ", position2)
                        return board, 0
    return board, 1


"""Handles the creation of a board. Performs turn-order checking
    and calls other functions for actions to be performed"""


def createBoard(moves):
    # Make sure correct color is moving.
    # Call movePiece or capturePiece.
    # Check if the piece ends at the end of the board, if so make it king.
    # 0 = empty
    # 1 = black checker
    # 2 = black king
    # 3 = white checker
    # 4 = white king
    # starting_board = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    board = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    # whose_turn: 0 = black, 1 = white
    whose_turn = 0
    errorcheck = 1

    if not (isinstance(moves, list)):
        # Does not have to be three, just meant to be a plain check for whether the moves are a valid set of moves.
        # Some move-data is a single integer, which is not enough to describe a board/match
        return board, 0

    for move in moves:
        checker_to_move = str(move[0])
        if (move[1].isdigit()):
            checker_to_move = checker_to_move + str(move[1])
        checker_to_move = int(checker_to_move)
        if (whose_turn == 0):
            if (board[checker_to_move - 1] == 1 or board[checker_to_move - 1] == 2):
                # turn order ok
                if (move.__contains__("-")):
                    board, errorecheck = movePiece(board, move, checker_to_move)
                elif (move.__contains__("x")):
                    board, errorcheck = capturePiece(board, move, checker_to_move)
                else:
                    print("Error 4, move does not contain - nor x")
                    return board, 0
                if (errorcheck == 0):
                    return board, 0
                whose_turn = 1

            else:
                print("Error 5, wrong turn order, Black's turn now, not White. Or trying to move non-existing piece: ",
                      board[checker_to_move - 1], " on position: ", checker_to_move)
                print("Moves: ", moves)
                print("Move: ", move)
                print(board)
                return board, 0
        elif (whose_turn == 1):
            if (board[checker_to_move - 1] == 3 or board[checker_to_move - 1] == 4):
                # turn order ok
                if (move.__contains__("-")):
                    board, errorcheck = movePiece(board, move, checker_to_move)
                elif (move.__contains__("x")):
                    board, errorcheck = capturePiece(board, move, checker_to_move)
                else:
                    print("Error 4, move does not contain - nor x")
                    return board, 0
                if (errorcheck == 0):
                    return board, 0
                whose_turn = 0
            else:
                print("Error 6, wrong turn order, White's turn now, not Black. Or trying to move non-existing piece: ",
                      board[checker_to_move - 1], " on position: ", checker_to_move)
                print("Moves: ", moves)
                print("Move: ", move)
                print(board)
                return board, 0

        else:
            print("Error 7, turn error in code")
            return board, 0

    return board, 1


def KFold_Maker(data, datasplits):
    losses = []
    wins = []
    draws = []
    for i in range(len(data)):

        if(len(data[i]) != 2):
            print("ERROR in KFold_Maker, length of data is not equal to 2, which means it does not contain both data and result")
            print("Skipping")
        elif (data[i][1] == 0):
            losses.append(data[i])
        elif (data[i][1] == 1):
            wins.append(data[i])
        elif (data[i][1] == 2):
            draws.append(data[i])
        else:
            print("ERROR in KFold_Maker, result not 0, 1 or 2")
            exit(69)
    print("Amount of losses: ", len(losses))
    print("Amount of wins: ", len(wins))
    print("Amount of draws: ", len(draws))

    # Datainfo stores information about the split of wins,losses and draws accross all datasets
    datainfo = []
    for y in range(datasplits):
        thisinfo = []
        for o in range(3):
            thisinfo.append(0)
        datainfo.append(thisinfo)

    traintestdatas = []
    for g in range(datasplits):
        somelist = []
        traintestdatas.append(somelist)
    splitwin = round_down(len(wins) * (datasplits / 100))
    splitloss = round_down(len(losses) * (datasplits / 100))
    splitdraw = round_down(len(draws) * (datasplits / 100))
    # ratio = round_down(len(wins)/len(losses))
    # testwin = len(wins) - trainwin
    # testloss = len(losses) - trainloss

    # Add win data
    changer = 0
    while True:
        if len(wins) <= 0:
            break
        if changer > datasplits - 1:
            changer = 0
        else:
            traintestdatas[changer].append(wins[0])
            del wins[0]
            datainfo[changer][0] += 1
            changer += 1

    # Add loss data
    changer = 0
    while True:
        if len(losses) <= 0:
            break
        if changer > datasplits - 1:
            changer = 0
        else:
            traintestdatas[changer].append(losses[0])
            del losses[0]
            datainfo[changer][1] += 1
            changer += 1

    # Add draw data
    changer = 0
    while True:
        if len(draws) <= 0:
            break
        if changer > datasplits - 1:
            changer = 0
        else:
            traintestdatas[changer].append(draws[0])
            del draws[0]
            datainfo[changer][2] += 1
            changer += 1

    # Shuffle each dataset individually
    for v in range(datasplits):
        np.random.shuffle(traintestdatas[v])

    for i in range(datasplits):
        print("Length of dataset number ", i, " is: ", len(traintestdatas[i]))

    # Combine the splits into dataplist(kfold amount) amount of datasets (train, test). Aka, make 10 sets of train and test datasets
    all_data = []
    for a in range(len(traintestdatas)):
        test_data = []
        for r in range(len(traintestdatas[a])):
            t = 0
            this_test = []
            for item in traintestdatas[a][r]:
                # appending list[moves, result]
                this_test.append(item)
                t += 1
            test_data.append(this_test)
        train_data = []
        for q in range(len(traintestdatas)):
            if q != a:
                for t in range(len(traintestdatas[q])):
                    train_data.append(traintestdatas[q][t])
        np.random.shuffle(test_data)
        np.random.shuffle(train_data)
        # So far so good, start from here....
        # We now have this kfold in the form train: train_data and test: test_data

        this_kfold = []
        this_kfold.append(train_data)
        this_kfold.append(test_data)
        all_data.append(this_kfold)

    return all_data


kfolds = KFold_Maker(extracted_data, 10)

# Create final board
# Create second final board
# Check for duplicates
# Store the information


"""Transform into pure binary (one board per type of piece per player, 4 boards in total, as well as adding the result"""


def toBinary(boards):
    print("Transforming boards to binary")
    binary_boards = []
    for i in range(len(boards)):
        if (len(boards[i][0]) != 32):
            print("Error in length of board, not equal to 32")
        this_binary_board = []
        for q in range(len(boards[i][0])):
            if (boards[i][0][q] == 1):
                this_binary_board.append(1)
            else:
                this_binary_board.append(0)
        for q in range(len(boards[i][0])):
            if (boards[i][0][q] == 2):
                this_binary_board.append(1)
            else:
                this_binary_board.append(0)
        for q in range(len(boards[i][0])):
            if (boards[i][0][q] == 3):
                this_binary_board.append(1)
            else:
                this_binary_board.append(0)
        for q in range(len(boards[i][0])):
            if (boards[i][0][q] == 4):
                this_binary_board.append(1)
            else:
                this_binary_board.append(0)
        this_binary_board.append(boards[i][1])
        binary_boards.append(this_binary_board)
    return binary_boards

"""
# Compare internally within each list, and between both lists
def removeKFoldDuplicates(train, test):
    print("Removing KFold Duplicates")
    no_duplicate_train = []
    no_duplicate_test = []
    trloss = 0
    trwin = 0
    trdraw = 0
    for q in range(len(train)):
        unique = 1
        for w in range(len(train)):
            if (q != w and train[q] == train[w]):
                unique = 0
        if (unique == 1):
            no_duplicate_train.append(train[q])
            if (train[q][-1] == 0):
                trloss += 1
            elif (train[q][-1] == 1):
                trwin += 1
            elif (train[q][-1] == 2):
                trdraw += 1
            else:
                print("ERROR 15, No result to record")
                print(train[q][-1])
                exit(15)
        else:
            if (not no_duplicate_train.__contains__(train[q])):
                no_duplicate_train.append(train[q])
                if (train[q][-1] == 0):
                    trloss += 1
                elif (train[q][-1] == 1):
                    trwin += 1
                elif (train[q][-1] == 2):
                    trdraw += 1
                else:
                    print("ERROR 15, No result to record")
                    exit(15)
    teloss = 0
    tewin = 0
    tedraw = 0
    for q in range(len(test)):
        unique = 1
        for w in range(len(test)):
            if (q != w and test[q] == [test[w]]):
                unique = 0
        if (unique == 1):
            no_duplicate_test.append(test[q])
            if (test[q][-1] == 0):
                teloss += 1
            elif (test[q][-1] == 1):
                tewin += 1
            elif (test[q][-1] == 2):
                tedraw += 1
            else:
                print("ERROR 15, No result to record")
                exit(15)
        else:
            if (not no_duplicate_test.__contains__(test[q])):
                no_duplicate_test.append(test[q])
                if (test[q][-1] == 0):
                    teloss += 1
                elif (test[q][-1] == 1):
                    tewin += 1
                elif (test[q][-1] == 2):
                    tedraw += 1
                else:
                    print("ERROR 15, No result to record")
                    exit(15)

    cross_no_duplicate_test = []
    teloss = 0
    tewin = 0
    tedraw = 0
    for q in range(len(no_duplicate_test)):
        unique = 1
        for w in range(len(no_duplicate_train)):
            if (no_duplicate_test[q] == no_duplicate_train[w]):
                unique = 0
        if (unique == 1):
            cross_no_duplicate_test.append(no_duplicate_test[q])
            if (test[q][-1] == 0):
                teloss += 1
            elif (test[q][-1] == 1):
                tewin += 1
            elif (test[q][-1] == 2):
                tedraw += 1
            else:
                print("ERROR 15, No result to record")
                exit(15)

    print("Train stats:")
    print("Amount of data: ", len(no_duplicate_train), " Losses: ", trloss, " Wins: ", trwin, " Draws: ", trdraw)
    print("Test stats:")
    print("Amount of data: ", len(cross_no_duplicate_test), " Losses: ", teloss, " Wins: ", tewin, " Draws: ", tedraw)

    return no_duplicate_train, cross_no_duplicate_test
"""

"""
print(len(kfolds))
print(len(kfolds[0]))           #Is in a kfold
print(len(kfolds[0][0]))        #Is in the train data
print(len(kfolds[0][1]))        #Is in the test data
print(len(kfolds[0][0][0]))     #Is in a datapoint
print(len(kfolds[0][0][1]))     #Is in another datapoint
print(kfolds[0][0][0][0])       #Shows a datapoint's moves
print(kfolds[0][0][0][1])       #Shows a datapoint's result
"""


# Sende inn et par færre moves i boardmaker for å få med de nest siste brettene også i tillegg til full moves.
# kjøre inn hvert sett i kfoldene i removeKFoldDuplicates
# Lagre hver kfold sett i filer
print("Creating boards")
for i in range(len(kfolds)):
    # Per kfold
    this_train = []
    this_test = []
    for q in range(len(kfolds[i])):
        # Per train or test
        these_boards = []
        for e in range(len(kfolds[i][q])):
            # Per board
            # Final board
            this_board = []
            created_board, errorcheck = createBoard(kfolds[i][q][e][0])
            if (errorcheck == 1):
                this_board.append(created_board)
                this_board.append(kfolds[i][q][e][1])
                these_boards.append(this_board)
            else:
                bad_boards += 1
                print("Bad board, total bad boards: ", bad_boards)

            """
            # Removed 1 turn (2 moves)
            this_board = []
            removed_one_move = kfolds[i][q][e][0][0:(len(kfolds[i][q][e][0]) - 2)]
            created_board, errorcheck = createBoard(removed_one_move)
            if (errorcheck == 1):
                this_board.append(created_board)
                this_board.append(kfolds[i][q][e][1])
                these_boards.append(this_board)
            else:
                bad_boards += 1
                print("Bad board, total bad boards: ", bad_boards)

            # Removed 2 turns (4 moves)
            this_board = []
            removed_two_moves = kfolds[i][q][e][0][0:(len(kfolds[i][q][e][0]) - 4)]
            created_board, errorcheck = createBoard(removed_two_moves)
            if (errorcheck == 1):
                this_board.append(created_board)
                this_board.append(kfolds[i][q][e][1])
                these_boards.append(this_board)
            else:
                bad_boards += 1
                print("Bad board, total bad boards: ", bad_boards)
            """


        these_binary = toBinary(these_boards)
        if (q == 0):
            this_train.append(these_binary)
        elif (q == 1):
            this_test.append(these_binary)
        else:
            print("Error, not train nor test")
            exit(1)
    # Sende inn til removeKFoldDuplicates

    # I messed up code somoewhere, so the whole list is put into this list. This line fixes this.
    this_train = this_train[0]
    this_test = this_test[0]

    #unique_train, unique_test = removeKFoldDuplicates(this_train, this_test)

    np.random.shuffle(this_train)
    np.random.shuffle(this_test)

    stringnumber = str(i)
    filename = stringnumber + "trainstatickfold.data"
    trainfile = open(filename, 'w')
    for r in range(len(this_train)):
        t = 0
        for item in this_train[r]:
            if t < len(this_train[r]) - 1:
                trainfile.write("%s" % item)
                trainfile.write(",")
            else:
                trainfile.write("%s" % item)
            t += 1
        trainfile.write("\n")
    trainfile.close()

    stringnumber = str(i)
    filename = stringnumber + "teststatickfold.data"
    testfile = open(filename, 'w')

    for r in range(len(this_test)):
        t = 0
        for item in this_test[r]:
            if t < len(this_test[r]) - 1:
                testfile.write("%s" % item)
                testfile.write(",")
            else:
                testfile.write("%s" % item)

            t += 1
        testfile.write("\n")
    testfile.close()

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

