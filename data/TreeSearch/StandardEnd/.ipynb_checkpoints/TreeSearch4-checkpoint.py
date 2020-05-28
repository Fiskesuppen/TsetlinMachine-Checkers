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


# Evaluates the board, returns loss 0, win 1 or draw 0.5
def evaluate(board, tm):
    # tm is either tmblack or tmwhite. tmblack evaluates the winning chance of black player. tmwhite evaluates the winning chance of white player.
    board = toBinary(board)
    check_board = []
    check_board.append(board)
    check_board = np.asarray(check_board)
    result = tm.predict(check_board)
    # Draw returns 0.5, a value or score between loss and win.
    # Draw
    if (result == 2):
        return 0.5
    # Win
    elif (result == 1):
        return 1
    # Loss
    elif (result == 0):
        return 0
    else:
        print("Result in evaluate not 2, 1 nor 0. This should not happen")
        exit(69)


#Evaluates the board, returns loss 0, win 1 or draw 0.5
def multiEvaluate(boards, tm):
    #tm is either tmblack or tmwhite. tmblack evaluates the winning chance of black player. tmwhite evaluates the winning chance of white player.
    check_boards = np.asarray(boards)
    result = tm.predict(check_boards)
    #Draw returns 0.5, a value or score between loss and win.
    results = []
    for i in range(len(result)):
        # Draw
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

print(" ")

import itertools
import random


"""
board = [4,0,1,0,0,0,0,3,1,0,0,3,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,0,0,3,3]
board = [1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,3,3,3,3,3,3,3,3,3,3,3,3]
result = evaluate(board, tmblack)
print(result)
result = evaluate(board, tmwhite)
print(result)
"""


def getDiagonalTable():
    diagonals = [[5, 6], [6, 7], [7, 8], [8], [1, 9], [1, 2, 9, 10], [2, 3, 10, 11],
                 [3, 4, 11, 12], [5, 6, 13, 14], [6, 7, 14, 15], [7, 8, 15, 16], [8, 16],
                 [9, 17], [9, 10, 17, 18], [10, 11, 18, 19], [11, 12, 19, 20],
                 [13, 14, 21, 22], [14, 15, 22, 23], [15, 16, 23, 24], [16, 24],
                 [17, 25], [17, 18, 25, 26], [18, 19, 26, 27], [19, 20, 27, 28],
                 [21, 22, 29, 30], [22, 23, 30, 31], [23, 24, 31, 32], [24, 32],
                 [25], [25, 26], [26, 27], [27, 28]]
    return diagonals


# Is position2 on the left of position1?
def isLeftSide(position1, position2):
    leftsidetable = [[5, 12, 21, 29], [5, 12, 21, 29, 1, 9, 17, 25, 6, 14, 22, 30],
                     [5, 12, 21, 29, 1, 9, 17, 25, 6, 14, 22, 30, 2, 10, 18, 26, 7, 15, 23, 31],
                     [5, 12, 21, 29, 1, 9, 17, 25, 6, 14, 22, 30, 2, 10, 18, 26, 7, 15, 23, 31, 3, 11, 19, 27, 8, 16,
                      24, 32],
                     [0], [5, 13, 21, 29, 1, 9, 17, 25], [5, 13, 21, 29, 1, 9, 17, 25, 6, 14, 22, 30, 2, 10, 18, 26],
                     [5, 13, 21, 29, 1, 9, 17, 25, 6, 14, 22, 30, 2, 10, 18, 26, 7, 15, 23, 31, 3, 11, 19, 27],
                     [5, 12, 21, 29], [5, 12, 21, 29, 1, 9, 17, 25, 6, 14, 22, 30],
                     [5, 12, 21, 29, 1, 9, 17, 25, 6, 14, 22, 30, 2, 10, 18, 26, 7, 15, 23, 31],
                     [5, 12, 21, 29, 1, 9, 17, 25, 6, 14, 22, 30, 2, 10, 18, 26, 7, 15, 23, 31, 3, 11, 19, 27, 8, 16,
                      24, 32],
                     [0], [5, 13, 21, 29, 1, 9, 17, 25], [5, 13, 21, 29, 1, 9, 17, 25, 6, 14, 22, 30, 2, 10, 18, 26],
                     [5, 13, 21, 29, 1, 9, 17, 25, 6, 14, 22, 30, 2, 10, 18, 26, 7, 15, 23, 31, 3, 11, 19, 27],
                     [5, 12, 21, 29], [5, 12, 21, 29, 1, 9, 17, 25, 6, 14, 22, 30],
                     [5, 12, 21, 29, 1, 9, 17, 25, 6, 14, 22, 30, 2, 10, 18, 26, 7, 15, 23, 31],
                     [5, 12, 21, 29, 1, 9, 17, 25, 6, 14, 22, 30, 2, 10, 18, 26, 7, 15, 23, 31, 3, 11, 19, 27, 8, 16,
                      24, 32],
                     [0], [5, 13, 21, 29, 1, 9, 17, 25], [5, 13, 21, 29, 1, 9, 17, 25, 6, 14, 22, 30, 2, 10, 18, 26],
                     [5, 13, 21, 29, 1, 9, 17, 25, 6, 14, 22, 30, 2, 10, 18, 26, 7, 15, 23, 31, 3, 11, 19, 27],
                     [5, 12, 21, 29], [5, 12, 21, 29, 1, 9, 17, 25, 6, 14, 22, 30],
                     [5, 12, 21, 29, 1, 9, 17, 25, 6, 14, 22, 30, 2, 10, 18, 26, 7, 15, 23, 31],
                     [5, 12, 21, 29, 1, 9, 17, 25, 6, 14, 22, 30, 2, 10, 18, 26, 7, 15, 23, 31, 3, 11, 19, 27, 8, 16,
                      24, 32],
                     [0], [5, 13, 21, 29, 1, 9, 17, 25], [5, 13, 21, 29, 1, 9, 17, 25, 6, 14, 22, 30, 2, 10, 18, 26],
                     [5, 13, 21, 29, 1, 9, 17, 25, 6, 14, 22, 30, 2, 10, 18, 26, 7, 15, 23, 31, 3, 11, 19, 27]]
    if (position2 in leftsidetable[position1 - 1]):
        return True
    else:
        return False


# Is position2 on the right of position1?
def isRightSide(position1, position2):
    rightsidetable = [[4, 12, 20, 28, 8, 16, 24, 32, 3, 11, 19, 27, 7, 15, 23, 31, 2, 10, 18, 26, 6, 14, 22, 30],
                      [4, 12, 20, 28, 8, 16, 24, 32, 3, 11, 19, 27, 7, 15, 23, 31], [4, 12, 20, 28, 8, 16, 24, 32], [0],
                      [4, 12, 20, 28, 8, 16, 24, 32, 3, 11, 19, 27, 7, 15, 23, 31, 2, 10, 18, 26, 6, 14, 22, 30, 1, 9,
                       17, 25], [4, 12, 20, 28, 8, 16, 24, 32, 3, 11, 19, 27, 7, 15, 23, 31, 2, 10, 18, 26],
                      [4, 12, 20, 28, 8, 16, 24, 32, 3, 11, 19, 27], [4, 12, 20, 28],
                      [4, 12, 20, 28, 8, 16, 24, 32, 3, 11, 19, 27, 7, 15, 23, 31, 2, 10, 18, 26, 6, 14, 22, 30],
                      [4, 12, 20, 28, 8, 16, 24, 32, 3, 11, 19, 27, 7, 15, 23, 31], [4, 12, 20, 28, 8, 16, 24, 32], [0],
                      [4, 12, 20, 28, 8, 16, 24, 32, 3, 11, 19, 27, 7, 15, 23, 31, 2, 10, 18, 26, 6, 14, 22, 30, 1, 9,
                       17, 25], [4, 12, 20, 28, 8, 16, 24, 32, 3, 11, 19, 27, 7, 15, 23, 31, 2, 10, 18, 26],
                      [4, 12, 20, 28, 8, 16, 24, 32, 3, 11, 19, 27], [4, 12, 20, 28],
                      [4, 12, 20, 28, 8, 16, 24, 32, 3, 11, 19, 27, 7, 15, 23, 31, 2, 10, 18, 26, 6, 14, 22, 30],
                      [4, 12, 20, 28, 8, 16, 24, 32, 3, 11, 19, 27, 7, 15, 23, 31], [4, 12, 20, 28, 8, 16, 24, 32], [0],
                      [4, 12, 20, 28, 8, 16, 24, 32, 3, 11, 19, 27, 7, 15, 23, 31, 2, 10, 18, 26, 6, 14, 22, 30, 1, 9,
                       17, 25], [4, 12, 20, 28, 8, 16, 24, 32, 3, 11, 19, 27, 7, 15, 23, 31, 2, 10, 18, 26],
                      [4, 12, 20, 28, 8, 16, 24, 32, 3, 11, 19, 27], [4, 12, 20, 28],
                      [4, 12, 20, 28, 8, 16, 24, 32, 3, 11, 19, 27, 7, 15, 23, 31, 2, 10, 18, 26, 6, 14, 22, 30],
                      [4, 12, 20, 28, 8, 16, 24, 32, 3, 11, 19, 27, 7, 15, 23, 31], [4, 12, 20, 28, 8, 16, 24, 32], [0],
                      [4, 12, 20, 28, 8, 16, 24, 32, 3, 11, 19, 27, 7, 15, 23, 31, 2, 10, 18, 26, 6, 14, 22, 30, 1, 9,
                       17, 25], [4, 12, 20, 28, 8, 16, 24, 32, 3, 11, 19, 27, 7, 15, 23, 31, 2, 10, 18, 26],
                      [4, 12, 20, 28, 8, 16, 24, 32, 3, 11, 19, 27], [4, 12, 20, 28]]
    if (position2 in rightsidetable[position1 - 1]):
        return True
    else:
        return False


def getSafePositions():
    safe = [1, 2, 3, 4, 5, 12, 13, 20, 21, 28, 29, 30, 31, 32]
    return safe


def isDiagonal(position1, position2):
    local_diagonals = getDiagonalTable()
    if (position2 in local_diagonals[position1 - 1] and position1 != 0 and position2 != 0):
        return True
    else:
        return False


"""
Gives a list of possible directions a piece may move. It makes sure pawns moves the right way.
"""


def getPossibleDirections(piece, diagonals, piece_type):
    possible_directions = []  # [position1, position2, position3....], each of them does not start at 0, is actual board number from 1
    # Checks for directions
    for i in range(len(diagonals[piece])):
        if (piece_type == 1):
            if (piece + 1 < diagonals[piece][i]):
                possible_directions.append(diagonals[piece][i])
        elif (piece_type == 3):
            if (piece + 1 > diagonals[piece][i]):
                possible_directions.append(diagonals[piece][i])
        elif (piece_type == 2 or piece_type == 4):
            possible_directions.append(diagonals[piece][i])
    return possible_directions


"""
Handles captures from a single position. Is used recursively. May return empty boards.
"""


def doCaptures(board, piece, piece_type, ally_pieces, enemy_pieces, diagonals, possible_directions):
    possible_captures = []
    # Check for possible captures, if found; only include captures
    for i in range(len(possible_directions)):
        # Check if there is an enemy in sight
        if (board[possible_directions[i] - 1] in enemy_pieces):
            # Check if enemy can be captured
            safe = getSafePositions()
            if (possible_directions[i] not in safe):
                # The piece is not near edge, it might be possible to capture
                # Find position for the piece to land on after capture
                # This handles right side
                if (isRightSide(piece + 1, possible_directions[i])):
                    # If the piece to capture is on the top-right diagonal
                    if (piece + 1 > possible_directions[i]):
                        # Check if this space is free
                        if (board[diagonals[possible_directions[i] - 1][1] - 1] == 0):
                            """
                            print("haha")
                            printFinalBoard(board)
                            print(piece)
                            print(diagonals[possible_directions[i] - 1][1] - 1)
                            print(possible_directions[i])
                            exit(22)
                            """
                            # Update a temporal version of the board
                            this_board = board.copy()
                            this_board[piece] = 0
                            this_board[diagonals[possible_directions[i] - 1][1] - 1] = piece_type
                            this_board[possible_directions[i] - 1] = 0
                            this_possible_directions = getPossibleDirections(
                                diagonals[possible_directions[i] - 1][1] - 1,
                                diagonals, piece_type)
                            # Check if more captures are possible
                            morecaptures = doCaptures(this_board, diagonals[possible_directions[i] - 1][1] - 1,
                                                      piece_type,
                                                      ally_pieces, enemy_pieces, diagonals, this_possible_directions)
                            if (len(morecaptures) != 0):
                                # Finalized captures
                                for r in range(len(morecaptures)):
                                    possible_captures.append(morecaptures[r])
                            else:
                                # This capture is already finalized, store this move instead
                                possible_captures.append(this_board)


                    # If the piece to capture is on the bottom-right diagonal
                    elif (piece + 1 < possible_directions[i]):
                        # Check if this space is free
                        # print(possible_directions[i] - 1)                #the piece to capture
                        # print(diagonals[possible_directions[i] - 1][3] -1)  #the destination
                        if (board[diagonals[possible_directions[i] - 1][3] - 1] == 0):
                            # Update a temporal version of the board
                            this_board = board.copy()
                            this_board[piece] = 0
                            this_board[diagonals[possible_directions[i] - 1][3] - 1] = piece_type
                            this_board[possible_directions[i] - 1] = 0
                            this_possible_directions = getPossibleDirections(
                                diagonals[possible_directions[i] - 1][3] - 1,
                                diagonals, piece_type)
                            # Check if more captures are possible
                            morecaptures = doCaptures(this_board, diagonals[possible_directions[i] - 1][3] - 1,
                                                      piece_type,
                                                      ally_pieces, enemy_pieces, diagonals, this_possible_directions)
                            if (len(morecaptures) != 0):
                                # Finalized captures
                                for r in range(len(morecaptures)):
                                    possible_captures.append(morecaptures[r])
                            else:
                                # This capture is already finalized, store this move instead
                                possible_captures.append(this_board)

                    else:
                        print(
                            "In doPiece: Piece to capture is the same as the piece to perform the capture. This should never happen.")

                # This handles left side
                elif (isLeftSide(piece + 1, possible_directions[i])):
                    # If the piece to capture is on the top-left diagonal
                    if (piece + 1 > possible_directions[i]):
                        # Check if this space is free
                        if (board[diagonals[possible_directions[i] - 1][0] - 1] == 0):
                            this_board = board.copy()
                            this_board[piece] = 0
                            this_board[diagonals[possible_directions[i] - 1][0] - 1] = piece_type
                            this_board[possible_directions[i] - 1] = 0
                            this_possible_directions = getPossibleDirections(
                                diagonals[possible_directions[i] - 1][0] - 1,
                                diagonals, piece_type)
                            # Check if more captures are possible
                            morecaptures = doCaptures(this_board, diagonals[possible_directions[i] - 1][0] - 1,
                                                      piece_type,
                                                      ally_pieces, enemy_pieces, diagonals, this_possible_directions)
                            if (len(morecaptures) != 0):
                                # Finalized captures
                                for r in range(len(morecaptures)):
                                    possible_captures.append(morecaptures[r])
                            else:
                                # This capture is already finalized, store this move instead
                                possible_captures.append(this_board)


                    # If the piece to capture is on the bottom-left diagonal
                    elif (piece + 1 < possible_directions[i]):
                        # Check if this space is free
                        if (board[diagonals[possible_directions[i] - 1][2] - 1] == 0):
                            this_board = board.copy()
                            this_board[piece] = 0
                            this_board[diagonals[possible_directions[i] - 1][2] - 1] = piece_type
                            this_board[possible_directions[i] - 1] = 0
                            this_possible_directions = getPossibleDirections(
                                diagonals[possible_directions[i] - 1][2] - 1,
                                diagonals, piece_type)
                            # Check if more captures are possible
                            morecaptures = doCaptures(this_board, diagonals[possible_directions[i] - 1][2] - 1,
                                                      piece_type,
                                                      ally_pieces, enemy_pieces, diagonals, this_possible_directions)
                            if (len(morecaptures) != 0):
                                # Finalized captures
                                for r in range(len(morecaptures)):
                                    possible_captures.append(morecaptures[r])
                            else:
                                # This capture is already finalized, store this move instead
                                possible_captures.append(this_board)
                    else:
                        print(
                            "In doPiece: Piece to capture is the same as the piece to perform the capture. This should never happen.")

    return possible_captures


def pawnToKing(boards):
    if (len(boards) != 0):
        for i in range(len(boards)):
            # White pawns to white kings
            if (boards[i][0] == 3):
                boards[i][0] = 4
            if (boards[i][1] == 3):
                boards[i][1] = 4
            if (boards[i][2] == 3):
                boards[i][2] = 4
            if (boards[i][3] == 3):
                boards[i][3] = 4

            # Black pawns to Black kings   31 30 29 28
            if (boards[i][28] == 1):
                boards[i][28] = 2
            if (boards[i][29] == 1):
                boards[i][29] = 2
            if (boards[i][30] == 1):
                boards[i][30] = 2
            if (boards[i][31] == 1):
                boards[i][31] = 2
    return boards


"""
Handles the movement of a specific piece. May return empty boards.
board = the board in list for
piece = the position of the piece in board. It is in list for, which meanse it starts at 0
"""


def doPiece(board, piece):
    # Check color
    # Check if capture is possible
    # If capture is possible, all capture moves must be performed/returned. Once starting a capture move, it must capture all the way.
    # No pure non-capture moves can be performed.
    # Else, return possible moves.
    piece_type = board[piece]
    ally_pieces = []
    enemy_pieces = []
    if (piece_type == 1 or piece_type == 2):
        ally_pieces = [1, 2]
        enemy_pieces = [3, 4]
    elif (piece_type == 3 or piece_type == 4):
        ally_pieces = [3, 4]
        enemy_pieces = [1, 2]
    diagonals = getDiagonalTable()

    # Checks for directions
    possible_directions = getPossibleDirections(piece, diagonals,
                                                piece_type)  # [position1, position2, position3....], each of them does not start at 0, is actual board number from 1

    possible_captures = doCaptures(board, piece, piece_type, ally_pieces, enemy_pieces, diagonals, possible_directions)

    # No captures possible, finding possible moves not involving captures
    if (len(possible_captures) == 0):
        possible_moves = []
        for o in range(len(possible_directions)):
            if (board[possible_directions[o] - 1] == 0):
                this_board = board.copy()
                this_board[piece] = 0
                this_board[possible_directions[o] - 1] = piece_type
                possible_moves.append(this_board)
        possible_moves = pawnToKing(possible_moves)
        return possible_moves, False

    # Captures possible, must perform one of these
    else:
        possible_captures = pawnToKing(possible_captures)
        return possible_captures, True


def calculateScores(boards):
    for i in range(len(boards)):
        avgscore = 0
        if (type(boards[i][1]) is list):
            for q in range(len(boards[i][1])):
                # boards[i][1][q][1] either a score or a list.
                # Make all of them have a score
                if (type(boards[i][1][q][1]) is list):
                    boards[i][1][q][1] = sum(boards[i][1][q][1]) / len(boards[i][1][q][1])

            # Sum up final
            for q in range(len(boards[i][1])):
                avgscore = avgscore + boards[i][1][q][1]
            avgscore = avgscore / len(boards[i][1])
            boards[i][1] = avgscore
    return boards


# first_boards[i][1][q]


# ALERT: 9 and -9 for guaranteed win and guaranteed loss is set to just a normal win or loss instead
# Player 1 is black and always starts
def player1DoTurn(board):
    # The goal of this function is to return a board where player 1/black has performed one move.
    # The decision is made by going through possible moves this turn,
    #   For each of these; go through each move white can do.
    #         For each of these; go through each move black can do.
    #              For each of these; evaluate and summarize all the way to the top to find what move gives the best chance of success
    # Pick the best move

    # Scores:
    # loss = 0
    # draw = 0.5
    # win = 1
    # guaranteed win = 9

    #print("Starting Player 1 first boards")

    first_boards = []  # Contains all boards black can make this turn.
    first_boards_capturemoves = []
    first_boards_onlymoves = []
    for i in range(len(board)):
        # find possible moves and append those moves
        # Capture moves override other moves
        # Check if the piece actually is owned by player 1 (black):
        if (board[i] == 1 or board[i] == 2):
            dopiece_boards, iscapture = doPiece(board, i)
            if (len(dopiece_boards) != 0):
                if (iscapture):
                    for z in range(len(dopiece_boards)):
                        first_boards_capturemoves.append(dopiece_boards[z])
                else:
                    for z in range(len(dopiece_boards)):
                        first_boards_onlymoves.append(dopiece_boards[z])

    # If any capture moves are possible, then only capture moves will be considered possible to perform due to the forced capture rule
    if (len(first_boards_capturemoves) != 0):
        for i in range(len(first_boards_capturemoves)):
            first_boards.append(first_boards_capturemoves[i])
    # If no capture moves are possible, consider the non-capture moves instead
    elif (len(first_boards_onlymoves) != 0):
        for i in range(len(first_boards_onlymoves)):
            first_boards.append(first_boards_onlymoves[i])
    # If no moves are possible to perform, we lost the game
    else:
        # We cannot perform any moves, We LOST!
        return board, True

    # [Depth one finished] first_boards done. Now, go into each of them and find possible boards white can make from these in one turn.

    #print("Starting Player 1 second boards")

    for i in range(len(first_boards)):
        second_boards_capturemoves = []
        second_boards_onlymoves = []
        for r in range(len(first_boards[i])):
            # find possible moves and append those moves
            # Capture moves override other moves
            # Check if the piece actually is owned by player 2 (white):
            if (first_boards[i][r] == 3 or first_boards[i][r] == 4):
                dopiece_boards, iscapture = doPiece(first_boards[i], r)
                if (len(dopiece_boards) != 0):
                    if (iscapture):
                        for z in range(len(dopiece_boards)):
                            second_boards_capturemoves.append(dopiece_boards[z])
                    else:
                        for z in range(len(dopiece_boards)):
                            second_boards_onlymoves.append(dopiece_boards[z])
        # If any capture moves are possible, then only capture moves will be considered possible to perform due to the forced capture rule
        if (len(second_boards_capturemoves) != 0):
            this_board = []
            this_board.append(first_boards[i])
            this_board.append(second_boards_capturemoves)
            first_boards[i] = this_board
        # If no capture moves are possible, consider the non-capture moves instead
        elif (len(second_boards_onlymoves) != 0):
            this_board = []
            this_board.append(first_boards[i])
            this_board.append(second_boards_onlymoves)
            first_boards[i] = this_board
        # No moves possible for black. This means that white wins with this move!
        else:
            # first_boards[i] gives a GUARANTEED WIN!!!
            # must find a way to give this a very high score
            win_board = []
            win_board.append(first_boards[i])
            win_board.append(1)
            first_boards[i] = win_board

    # [Depth two finished]

    #print("Starting Player 1 third boards")
    
    for i in range(len(first_boards)):
        if (type(first_boards[i][1]) is list):
            for q in range(len(first_boards[i][1])):
                # first_boards[i][1][q] is a board that must be iterated through to find possible moves and create new boards.
                third_boards_capturemoves = []
                third_boards_onlymoves = []
                for r in range(len(first_boards[i][1][q])):
                    # first_boards[i][1][q][r] is a specific piece in a board
                    if (first_boards[i][1][q][r] == 1 or first_boards[i][1][q][r] == 2):
                        dopiece_boards, iscapture = doPiece(first_boards[i][1][q], r)
                        if (len(dopiece_boards) != 0):
                            if (iscapture):
                                for z in range(len(dopiece_boards)):
                                    third_boards_capturemoves.append(dopiece_boards[z])
                            else:
                                for z in range(len(dopiece_boards)):
                                    third_boards_onlymoves.append(dopiece_boards[z])

                # If any capture moves are possible, then only capture moves will be considered possible to perform due to the forced capture rule
                if (len(third_boards_capturemoves) != 0):
                    this_board = []
                    this_board.append(first_boards[i][1][q])
                    bin_boards = []
                    for o in range(len(third_boards_capturemoves)):
                        bin_boards.append(toBinary(third_boards_capturemoves[o]))
                    results = multiEvaluate(bin_boards, tmblack)
                    for o in range(len(third_boards_capturemoves)):
                        third_boards_capturemoves[o] = results[o]
                    this_board.append(third_boards_capturemoves)
                    first_boards[i][1][q] = this_board
                # If no capture moves are possible, consider the non-capture moves instead
                elif (len(third_boards_onlymoves) != 0):
                    this_board = []
                    this_board.append(first_boards[i][1][q])
                    bin_boards = []
                    for o in range(len(third_boards_onlymoves)):
                        bin_boards.append(toBinary(third_boards_onlymoves[o]))
                    results = multiEvaluate(bin_boards, tmblack)
                    for o in range(len(third_boards_onlymoves)):
                        third_boards_onlymoves[o] = results[o]
                    this_board.append(third_boards_onlymoves)
                    first_boards[i][1][q] = this_board
                # No moves possible for black. This means that white wins with this move!
                else:
                    # first_boards[i][1][q] gives a GUARANTEED LOSS....
                    # Picking move first_boards[i] allows the opponent to pick a move that leads to this loss. We really do not want to take that chance.
                    # Maybe find a way to give this a very low score.
                    loss_board = []
                    loss_board.append(first_boards[i][1][q])
                    loss_board.append(0)
                    first_boards[i][1][q] = loss_board

    # Summarization of results
    # Functionality for going from the bottom to the top of the list to end up with a list of moves with corresponding probabilities of win.
    # scores:
    #          guaranteed win after 1 move is given a score of 9 as this guarantees victory
    #          Win score modifier: 1
    #          Draw score modifier: 0.5   #This is for making draws more desireable than losses.
    #          Loss score is kept at 0

    #print("Player 1 Calculating scores")

    first_boards = calculateScores(first_boards)
    
            
    #print("Player 1 finding highest score of move")
    highest = first_boards[0]
    for i in range(len(first_boards)):
        if (first_boards[i][1] > highest[1]):
            highest = first_boards[i]
            
    #Trying to be a bit unpredictable by choosing random from the best moves if there is a tie.
    highests = []            
    for i in range(len(first_boards)):
        if (first_boards[i][1] == highest[1]):
            highests.append(first_boards[i][0])
    move_to_pick = random.randint(0, (len(highests)-1))
    
    return highests[move_to_pick], False


# Player 2 is white and always starts
def player2DoTurn(board):
    # The goal of this function is to return a board where player 2/white has performed one move.
    # The decision is made by going through possible moves this turn,
    #   For each of these; go through each move black can do.
    #         For each of these; go through each move white can do.
    #              For each of these; evaluate and summarize all the way to the top to find what move gives the best chance of success
    # Pick the best move

    # Scores:
    # loss = 0
    # draw = 0.5
    # win = 1
    # guaranteed win = 9

    #print("Starting Player 2 first boards")

    first_boards = []  # Contains all boards white can make this turn.
    first_boards_capturemoves = []
    first_boards_onlymoves = []
    for i in range(len(board)):
        # find possible moves and append those moves
        # Capture moves override other moves
        # Check if the piece actually is owned by player 2 (white):
        if (board[i] == 3 or board[i] == 4):
            dopiece_boards, iscapture = doPiece(board, i)
            if (len(dopiece_boards) != 0):
                if (iscapture):
                    for z in range(len(dopiece_boards)):
                        first_boards_capturemoves.append(dopiece_boards[z])
                else:
                    for z in range(len(dopiece_boards)):
                        first_boards_onlymoves.append(dopiece_boards[z])
    # If any capture moves are possible, then only capture moves will be considered possible to perform due to the forced capture rule
    if (len(first_boards_capturemoves) != 0):
        for i in range(len(first_boards_capturemoves)):
            first_boards.append(first_boards_capturemoves[i])
    # If no capture moves are possible, consider the non-capture moves instead
    elif (len(first_boards_onlymoves) != 0):
        for i in range(len(first_boards_onlymoves)):
            first_boards.append(first_boards_onlymoves[i])
    # If no moves are possible to perform, we lost the game
    else:
        # We cannot perform any moves, We LOST!
        return board, True

    # [Depth one finished] first_boards done. Now, go into each of them and find possible boards black can make from these in one turn.

    #print("Starting Player 2 second boards")

    for i in range(len(first_boards)):
        second_boards_capturemoves = []
        second_boards_onlymoves = []

        for r in range(len(first_boards[i])):
            # find possible moves and append those moves
            # Capture moves override other moves
            # Check if the piece actually is owned by player 1 (black):
            if (first_boards[i][r] == 1 or first_boards[i][r] == 2):

                dopiece_boards, iscapture = doPiece(first_boards[i], r)
                if (len(dopiece_boards) != 0):
                    if (iscapture):
                        for z in range(len(dopiece_boards)):
                            second_boards_capturemoves.append(dopiece_boards[z])
                    else:
                        for z in range(len(dopiece_boards)):
                            second_boards_onlymoves.append(dopiece_boards[z])

        # If any capture moves are possible, then only capture moves will be considered possible to perform due to the forced capture rule
        if (len(second_boards_capturemoves) != 0):
            this_board = []
            this_board.append(first_boards[i])
            this_board.append(second_boards_capturemoves)
            first_boards[i] = this_board
        # If no capture moves are possible, consider the non-capture moves instead
        elif (len(second_boards_onlymoves) != 0):
            this_board = []
            this_board.append(first_boards[i])
            this_board.append(second_boards_onlymoves)
            first_boards[i] = this_board
        # No moves possible for white. This means that black wins with this move!
        else:
            # first_boards[i] gives a GUARANTEED WIN!!!
            # must find a way to give this a very high score
            win_board = []
            win_board.append(first_boards[i])
            win_board.append(9)
            first_boards[i] = win_board

    # [Depth two finished]

    #print("Starting Player 2 third boards")
    
    for i in range(len(first_boards)):
        if (type(first_boards[i][1]) is list):
            for q in range(len(first_boards[i][1])):
                # first_boards[i][1][q] is a board that must be iterated through to find possible moves and create new boards.
                third_boards_capturemoves = []
                third_boards_onlymoves = []
                for r in range(len(first_boards[i][1][q])):
                    # first_boards[i][1][q][r] is a specific piece in a board
                    if (first_boards[i][1][q][r] == 3 or first_boards[i][1][q][r] == 4):
                        dopiece_boards, iscapture = doPiece(first_boards[i][1][q], r)
                        if (len(dopiece_boards) != 0):
                            if (iscapture):
                                for z in range(len(dopiece_boards)):
                                    third_boards_capturemoves.append(dopiece_boards[z])
                            else:
                                for z in range(len(dopiece_boards)):
                                    third_boards_onlymoves.append(dopiece_boards[z])

                # If any capture moves are possible, then only capture moves will be considered possible to perform due to the forced capture rule
                if (len(third_boards_capturemoves) != 0):
                    this_board = []
                    this_board.append(first_boards[i][1][q])
                    bin_boards = []
                    for o in range(len(third_boards_capturemoves)):
                        bin_boards.append(toBinary(third_boards_capturemoves[o]))
                    results = multiEvaluate(bin_boards, tmwhite)
                    for o in range(len(third_boards_capturemoves)):
                        third_boards_capturemoves[o] = results[o]
                    this_board.append(third_boards_capturemoves)
                    first_boards[i][1][q] = this_board
                # If no capture moves are possible, consider the non-capture moves instead
                elif (len(third_boards_onlymoves) != 0):
                    this_board = []
                    this_board.append(first_boards[i][1][q])
                    bin_boards = []
                    for o in range(len(third_boards_onlymoves)):
                        bin_boards.append(toBinary(third_boards_onlymoves[o]))
                    results = multiEvaluate(bin_boards, tmwhite)
                    for o in range(len(third_boards_onlymoves)):
                        third_boards_onlymoves[o] = results[o]
                    this_board.append(third_boards_onlymoves)
                    first_boards[i][1][q] = this_board
                # No moves possible for white. This means that black wins with this move!
                else:
                    # first_boards[i][1][q] gives a GUARANTEED LOSS....
                    # Picking move first_boards[i] allows the opponent to pick a move that leads to this loss. We really do not want to take that chance.
                    # Maybe find a way to give this a very low score.
                    loss_board = []
                    loss_board.append(first_boards[i][1][q])
                    loss_board.append(-9)
                    first_boards[i][1][q] = loss_board

    # Summarization of results
    # Functionality for going from the bottom to the top of the list to end up with a list of moves with corresponding probabilities of win.
    # scores:
    #          guaranteed win after 1 move is given a score of 9 as this guarantees victory
    #          Win score modifier: 1
    #          Draw score modifier: 0.5   #This is for making draws more desireable than losses.
    #          Loss score is kept at 0

    #print("Player 2 Calculating scores")

    first_boards = calculateScores(first_boards)

    #print("Player 2 finding highest score of move")
    highest = first_boards[0]
    for i in range(len(first_boards)):
        if (first_boards[i][1] > highest[1]):
            highest = first_boards[i]
            
    #Trying to be a bit unpredictable by choosing random from the best moves if there is a tie.
    highests = []            
    for i in range(len(first_boards)):
        if (first_boards[i][1] == highest[1]):
            highests.append(first_boards[i][0])
    move_to_pick = random.randint(0, (len(highests)-1))
    
    return highests[move_to_pick], False



# Player 2 is white and always starts
def player2DoTurnStupid(board):
    # The goal of this function is to return a board where player 2/white has performed one move.
    # The decision is made by going through possible moves this turn,
    #   For each of these; go through each move black can do.
    #         For each of these; go through each move white can do.
    #              For each of these; evaluate and summarize all the way to the top to find what move gives the best chance of success
    # Pick the best move

    # Scores:
    # loss = 0
    # draw = 0.5
    # win = 1
    # guaranteed win = 9

    #print("Starting Player 2 first boards")

    first_boards = []  # Contains all boards white can make this turn.
    first_boards_capturemoves = []
    first_boards_onlymoves = []
    for i in range(len(board)):
        # find possible moves and append those moves
        # Capture moves override other moves
        # Check if the piece actually is owned by player 2 (white):
        if (board[i] == 3 or board[i] == 4):
            dopiece_boards, iscapture = doPiece(board, i)
            if (len(dopiece_boards) != 0):
                if (iscapture):
                    for z in range(len(dopiece_boards)):
                        first_boards_capturemoves.append(dopiece_boards[z])
                else:
                    for z in range(len(dopiece_boards)):
                        first_boards_onlymoves.append(dopiece_boards[z])
    # If any capture moves are possible, then only capture moves will be considered possible to perform due to the forced capture rule
    if (len(first_boards_capturemoves) != 0):
        for i in range(len(first_boards_capturemoves)):
            first_boards.append(first_boards_capturemoves[i])
    # If no capture moves are possible, consider the non-capture moves instead
    elif (len(first_boards_onlymoves) != 0):
        for i in range(len(first_boards_onlymoves)):
            first_boards.append(first_boards_onlymoves[i])
    # If no moves are possible to perform, we lost the game
    else:
        # We cannot perform any moves, We LOST!
        return board, True

    rand_move = random.randint(0, len(first_boards) -1)
    return first_boards[rand_move], False






def printFinalBoard(board):
    print(" ", board[0], " ", board[1], " ", board[2], " ", board[3])
    print(board[4], " ", board[5], " ", board[6], " ", board[7])
    print(" ", board[8], " ", board[9], " ", board[10], " ", board[11])
    print(board[12], " ", board[13], " ", board[14], " ", board[15])
    print(" ", board[16], " ", board[17], " ", board[18], " ", board[19])
    print(board[20], " ", board[21], " ", board[22], " ", board[23])
    print(" ", board[24], " ", board[25], " ", board[26], " ", board[27])
    print(board[28], " ", board[29], " ", board[30], " ", board[31])

    print(" ")


current_board = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]

# black player = player 1 = 1
# white player = player 2 = 2


# Tells whether the game is over or not. True = the game is going on right now.
active = True


printFinalBoard(current_board)

player1_wins = 0
player2_wins = 0
draws = 0

for i in range(100):
    current_board = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    gamestart_time = time()
    playerturn = 1
    turn_number = 0
    active = True
    while (active == True):

        #start_time = time()

        if (turn_number == 500):
            print("500 turns have passed, this match is probably a draw")
            printFinalBoard(current_board)
            draws += 1
            break
        turn_number += 1
        #print("Turn number ", turn_number)

        if (playerturn == 1):
            # Player 1 does their turn
            #print("Player 1's turn")
            current_board, lost = player1DoTurn(current_board)
            if (lost == True):
                print("Player 1 lost")
                printFinalBoard(current_board)
                player2_wins += 1
                active = False
                break
            playerturn = 2

        elif (playerturn == 2):
            # Player 2 does their turn
            #print("Player 2's turn")
            current_board, lost = player2DoTurn(current_board)
            if (lost == True):
                print("Player 2 lost")
                printFinalBoard(current_board)
                player1_wins += 1
                active = False
                break
            playerturn = 1

        #printFinalBoard(current_board)
        #stop_time = time()
        #print("Time elapsed this turn: " , round(stop_time - start_time, 2), " seconds")


    gameover_time = time()
    print("Ended at turn ", turn_number)
    print("Time elapsed this game: " , round(gameover_time - gamestart_time, 2), " seconds")
    print("This was match number: ", i + 1, " The standings are: Player 1 wins: ", player1_wins, " Player 2 wins: ", player2_wins, " Draws: ", draws)
    print(" ")


    