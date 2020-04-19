
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
    leftsidetable = [[5, 12, 21, 29], [5, 12, 21, 29, 1, 9, 17, 25, 6, 14, 22, 30], [5, 12, 21, 29, 1, 9, 17, 25, 6, 14, 22, 30, 2, 10, 18, 26, 7, 15, 23, 31], [5, 12, 21, 29, 1, 9, 17, 25, 6, 14, 22, 30, 2, 10, 18, 26, 7, 15, 23, 31, 3, 11, 19, 27, 8, 16, 24, 32],
                     [0], [5, 13, 21, 29, 1, 9, 17, 25], [5, 13, 21, 29, 1, 9, 17, 25, 6, 14, 22, 30, 2, 10, 18, 26], [5, 13, 21, 29, 1, 9, 17, 25, 6, 14, 22, 30, 2, 10, 18, 26, 7, 15, 23, 31, 3, 11, 19, 27],
                     [5, 12, 21, 29], [5, 12, 21, 29, 1, 9, 17, 25, 6, 14, 22, 30], [5, 12, 21, 29, 1, 9, 17, 25, 6, 14, 22, 30, 2, 10, 18, 26, 7, 15, 23, 31], [5, 12, 21, 29, 1, 9, 17, 25, 6, 14, 22, 30, 2, 10, 18, 26, 7, 15, 23, 31, 3, 11, 19, 27, 8, 16, 24, 32],
                     [0], [5, 13, 21, 29, 1, 9, 17, 25], [5, 13, 21, 29, 1, 9, 17, 25, 6, 14, 22, 30, 2, 10, 18, 26], [5, 13, 21, 29, 1, 9, 17, 25, 6, 14, 22, 30, 2, 10, 18, 26, 7, 15, 23, 31, 3, 11, 19, 27],
                     [5, 12, 21, 29], [5, 12, 21, 29, 1, 9, 17, 25, 6, 14, 22, 30], [5, 12, 21, 29, 1, 9, 17, 25, 6, 14, 22, 30, 2, 10, 18, 26, 7, 15, 23, 31], [5, 12, 21, 29, 1, 9, 17, 25, 6, 14, 22, 30, 2, 10, 18, 26, 7, 15, 23, 31, 3, 11, 19, 27, 8, 16, 24, 32],
                     [0], [5, 13, 21, 29, 1, 9, 17, 25], [5, 13, 21, 29, 1, 9, 17, 25, 6, 14, 22, 30, 2, 10, 18, 26], [5, 13, 21, 29, 1, 9, 17, 25, 6, 14, 22, 30, 2, 10, 18, 26, 7, 15, 23, 31, 3, 11, 19, 27],
                     [5, 12, 21, 29], [5, 12, 21, 29, 1, 9, 17, 25, 6, 14, 22, 30], [5, 12, 21, 29, 1, 9, 17, 25, 6, 14, 22, 30, 2, 10, 18, 26, 7, 15, 23, 31], [5, 12, 21, 29, 1, 9, 17, 25, 6, 14, 22, 30, 2, 10, 18, 26, 7, 15, 23, 31, 3, 11, 19, 27, 8, 16, 24, 32],
                     [0], [5, 13, 21, 29, 1, 9, 17, 25], [5, 13, 21, 29, 1, 9, 17, 25, 6, 14, 22, 30, 2, 10, 18, 26], [5, 13, 21, 29, 1, 9, 17, 25, 6, 14, 22, 30, 2, 10, 18, 26, 7, 15, 23, 31, 3, 11, 19, 27]]
    if (position2 in leftsidetable[position1 -1]):
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
    #print("Length of possible_directions = ", len(possible_directions))
    # Check for possible captures, if found; only include captures
    for i in range(len(possible_directions)):
        #print("Checking square")
        # Check if there is an enemy in sight
        #print("Checking out: ", possible_directions[i])
        if (board[possible_directions[i] -1] in enemy_pieces):
            #print("Enemy spotted")
            # Check if enemy can be captured
            safe = getSafePositions()
            if (possible_directions[i] not in safe):
                # The piece is not near edge, it might be possible to capture
                #print("The enemy is not safe")
                # Find position for the piece to land on after capture

                # This handles right side
                if (isRightSide(piece + 1, possible_directions[i])):
                    #print("The enemy is on the right side")
                    # If the piece to capture is on the top-right diagonal
                    if (piece + 1 > possible_directions[i]):
                        #print("The enemy is on the top-right diagonal")
                        # Check if this space is free
                        if (board[diagonals[possible_directions[i] - 1][1]-1] == 0):

                            # Update a temporal version of the board
                            this_board = board.copy()
                            this_board[piece] = 0
                            this_board[diagonals[possible_directions[i] - 1][1]-1] = piece_type
                            this_board[possible_directions[i] - 1] = 0
                            this_possible_directions = getPossibleDirections(diagonals[possible_directions[i] - 1][1]-1,
                                                                             diagonals, piece_type)
                            # Check if more captures are possible
                            morecaptures = doCaptures(this_board, diagonals[possible_directions[i] - 1][1]-1, piece_type,
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
                        #print("The enemy is on the bottom-right diagonal")
                        # Check if this space is free
                        #print(possible_directions[i] - 1)                #the piece to capture
                        #print(diagonals[possible_directions[i] - 1][3] -1)  #the destination
                        if (board[diagonals[possible_directions[i] - 1][3]-1] == 0):
                            #print("The space is free")
                            # Update a temporal version of the board
                            this_board = board.copy()
                            this_board[piece] = 0
                            this_board[diagonals[possible_directions[i] - 1][3]-1] = piece_type
                            this_board[possible_directions[i] - 1] = 0
                            this_possible_directions = getPossibleDirections(diagonals[possible_directions[i] - 1][3]-1,
                                                                             diagonals, piece_type)
                            # Check if more captures are possible
                            morecaptures = doCaptures(this_board, diagonals[possible_directions[i] - 1][3]-1, piece_type,
                                                      ally_pieces, enemy_pieces, diagonals, this_possible_directions)
                            if (len(morecaptures) != 0):
                                # Finalized captures
                                for r in range(len(morecaptures)):
                                    possible_captures.append(morecaptures[r])
                            else:
                                # This capture is already finalized, store this move instead
                                possible_captures.append(this_board)

                    else:
                        print("In doPiece: Piece to capture is the same as the piece to perform the capture. This should never happen.")

                # This handles left side
                elif (isLeftSide(piece + 1, possible_directions[i])):
                    # If the piece to capture is on the top-left diagonal
                    if (piece + 1 > possible_directions[i]):
                        # Check if this space is free
                        if (board[diagonals[possible_directions[i] - 1][0]-1] == 0):
                            this_board = board.copy()
                            this_board[piece] = 0
                            this_board[diagonals[possible_directions[i] - 1][0]-1] = piece_type
                            this_board[possible_directions[i] - 1] = 0
                            this_possible_directions = getPossibleDirections(diagonals[possible_directions[i] - 1][0]-1,
                                                                             diagonals, piece_type)
                            # Check if more captures are possible
                            morecaptures = doCaptures(this_board, diagonals[possible_directions[i] - 1][0]-1, piece_type,
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
                        if (board[diagonals[possible_directions[i] - 1][2]-1] == 0):
                            this_board = board.copy()
                            this_board[piece] = 0
                            this_board[diagonals[possible_directions[i] - 1][2]-1] = piece_type
                            this_board[possible_directions[i] - 1] = 0
                            this_possible_directions = getPossibleDirections(diagonals[possible_directions[i] - 1][2]-1,
                                                                             diagonals, piece_type)
                            # Check if more captures are possible
                            morecaptures = doCaptures(this_board, diagonals[possible_directions[i] - 1][2]-1, piece_type,
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


def printFinalBoard(board):
    print(" ", board[0], " ", board[1], " ", board[2], " ", board[3])
    print(board[4], " ", board[5], " ", board[6], " ", board[7])
    print(" ",board[8], " ", board[9], " ", board[10], " ", board[11])
    print(board[12], " ", board[13], " ", board[14], " ", board[15])
    print(" ",board[16], " ", board[17], " ", board[18], " ", board[19])
    print(board[20], " ", board[21], " ", board[22], " ", board[23])
    print(" ",board[24], " ", board[25], " ", board[26], " ", board[27])
    print(board[28], " ", board[29], " ", board[30], " ", board[31])

diagonals = getDiagonalTable()
"""
current_board = [1,1,1,1,
                 1,1,1,1,
                 1,1,1,1,
                 0,0,0,0,
                 0,0,0,0,
                 3,3,3,3,
                 3,3,3,3,
                 3,3,3,3]
"""
current_board =  [0,0,0,0,
                 0,0,0,0,
                  0,0,0,0,
                 0,0,0,0,
                  0,3,3,0,
                 0,0,0,2,
                  4,0,0,0,
                 0,0,0,0]




position = 24
piece = position -1



print(getPossibleDirections(piece, diagonals, current_board[piece]))

print("Orignal board:")
printFinalBoard(current_board)
print("\n")

#doCaptures(board, piece, piece_type, ally_pieces, enemy_pieces, diagonals, possible_directions):
final_boards = doCaptures(current_board, piece, current_board[piece], [1,2], [3,4], diagonals, getPossibleDirections(piece, diagonals, current_board[piece]))

if(len(final_boards) != 0):
    for i in range(len(final_boards)):
        if(len(final_boards[i]) != 0):
            printFinalBoard(final_boards[i])
            print('\n')

