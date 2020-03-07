import csv
import math

#my_data = np.genfromtxt('OCA_2.0.csv', delimiter=',')
"""Create the starting board and define what data represents what type of piece"""
print("Extracting moves")
#print(my_data)
#[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
#[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#0 = empty
#1 = black checker
#2 = black king
#3 = white checker
#4 = white king
#starting_board = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]

#print(len(starting_board))
#print(starting_board)


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



#lines = open("minidata.csv").read().splitlines()
lines = open("OCA_2.0.csv").read().splitlines()
#print(f.readlines())
datapoints = 0
fetch_moves = 0
extracted_data = []
this_result = 0
this_play = []

loaded_datapoints = 0
bad_data= 0

for i in range(len(lines)):
    #print(lines[i])
    if(lines[i].__contains__("Result")):
        if(lines[i].__contains__("0-1")):
            this_result = 0
            losses += 1
        elif(lines[i].__contains__("1-0")):
            this_result = 1
            wins += 1
        elif(lines[i].__contains__("1/2-1/2")):
            this_result = 2
            draws += 1
        else:
            print("ERROR 2 on number ", i, "undefined Result: ", lines[i])
            exit(2)
        fetch_moves = 1
        datapoints += 1
        #print("Result found: ", this_result)
    elif(fetch_moves != 0):
        #if(not(lines[i][0] == )):
        #print("Fetching moves")
        if(len(lines[i]) > 0):
            if(lines[i][0].isdigit()):
                #print("Adding move")
                this_play.append(lines[i])
        else:
            #print("Finalizing play")
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
                #exit(3)


            if(bad_data == 1):
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


#print("Extracted data: ")
#for i in range(len(extracted_data)):
    #print(extracted_data[i])

#print("Datapoints: ", datapoints)
#print("Loaded datapoints: ", loaded_datapoints)

#One full datapoint
print(extracted_data[0])
#One datapoint's list of moves
print(extracted_data[0][0])
#One datapoint's result
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



"""Build the boards, while checking for illegal moves as a way to counteract possible typing errors"""
#There are multiple ways to create data of this.
    #Must do
        #Create final board with the result
    #Maybe do
        #Create multiple boards with some result. Possibly average out equal boards in case equal boards may lead to
        #different results or simply allow "duplicates" with different results.
        #The power of Tsetlin machine may find out that three equal boards with
        #different outcomes states that the outcome is uncertain?
    #Granmo tips regarding data making:
        #You can create more data by using previous moves, BUT: make sure to keep the bonus data
        #in the same data category (training/test) in order to not contaminate the test data.

#In theory:
#[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
#In programming:
#[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
#[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#0 = empty
#1 = black checker
#2 = black king
#3 = white checker
#4 = white king
#starting_board = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]


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
    if(position2 in local_diagonals[position1 -1] and position1 != 0 and position2 != 0):
        return True
    else:
        return False

def isMovingRightWay(checker_to_move, moveto, piece_type, board):
    if(piece_type not in [1,2,3,4]):
        print("Error 10, trying to move a non-existing piece, piece is: ", piece_type, " location: ", checker_to_move)
        return False
    if(piece_type == 1):
        if(checker_to_move > moveto):
            print("Wrong way")
            return False
    elif(piece_type == 3):
        if(checker_to_move < moveto):
            print("Wrong way")
            return False
    return True


def movePiece(board, action, checker_to_move):
    moveto = str(action[action.index("-") + 1])
    if(action.index("-") + 2 < len(action)):
        moveto = moveto + str(action[action.index("-") + 2])
    moveto = int(moveto)
    if(board[moveto-1] == 0):
        if(isDiagonal(checker_to_move, moveto) and isMovingRightWay(checker_to_move, moveto, board[checker_to_move-1], board)):
            board[moveto-1]= board[checker_to_move-1]
            board[checker_to_move-1] = 0
            upper_border = [1, 2, 3, 4]
            lower_border = [29, 30, 31, 32]
            if(upper_border.__contains__(moveto) and board[moveto-1] == 3):
                board[moveto -1] = 4
            elif(lower_border.__contains__(moveto) and board[moveto-1] == 1):
                board[moveto -1] = 2
            return board, 1
        else:
            print("Error 9, trying to move piece to non-diagonal spot or wrong way, checker_to_move: ", checker_to_move, " moveto: ", moveto, " board[checker_to_move-1]: ", board[checker_to_move-1])
            return board, 0
    else:
        print("Error 8, trying to move piece to an occupied spot")
        return board, 0

def isCaptureDiagonal(position1, position2):
    position1 += 1
    position2 += 1
    if((position1-position2) == 9 or (position2-position1) == 9 or (position1-position2) == 7 or (position2-position1) == 7):
        return True
    else:
        return False


"""This function actually performs the capture action
    Returns board"""
def doCapture(board, position1, position2):
    #Checks if captures opposite color
    diagonals = getDiagonalTable()
    if((len(list(set(diagonals[position1 -1]).intersection(diagonals[position2 -1])))) == 1):
        to_capture = list(set(diagonals[position1 -1]).intersection(diagonals[position2 -1]))[0]
    else:
        print("Error 12, found none or multiple pieces to capture when comparing diagonals")
        return board, 0
    black_pieces = [1,2]
    white_pieces = [3,4]
    if(board[to_capture -1] == 0):
        print("Error 14, trying to capture blank position")
        return board, 0
    elif((board[position1-1] in black_pieces and board[to_capture -1] in black_pieces) or (board[position1-1] in white_pieces and board[to_capture -1] in white_pieces)):
        print("Error 13, trying to capture piece of same color")
        return board, 0
    else:
        board[position2 -1] = board[position1 -1]
        board[position1 -1] = 0
        board[to_capture -1] = 0
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
    #Make sure that the jump is ok:
        #Does jump over enemy piece
        #Is diagonal, with extra step
    if(isMovingRightWay(position1, position2, board[position1-1], board) and isCaptureDiagonal(position1,position2)):
        return True
    else:
        return False




"""This function is responsible for handling the captures of pieces.
    It will dissect all capture actions into single capture actions
    to send to isLegalCapture"""
def capturePiece(board, action, checker_to_move):
    #Create a custom function, isLegalCapture.
    position1 = checker_to_move
    first_position_switch = 0
    errorcheck = 1
    for i in range(len(action)):
        if(action[i].__contains__("x")):
            if((i+1) < len(action)):
                if(action[i+1].isdigit()):
                    if(first_position_switch == 1):
                        position1 = position2
                    position2 = str(action[i+1])
                    if((i+2) < len(action)):
                        if(action[i + 2].isdigit()):
                            position2 = position2 + str(action[i+2])
                    position2 = int(position2)
                    if(isLegalCapture(board, position1, position2)):
                        board, errorcheck = doCapture(board, position1, position2)
                        if(errorcheck == 0):
                            return board, 0
                        first_position_switch = 1
                    else:
                        print("Error 11, illegal capture attempted, wrong direction or not proper capture diagonal, action: ", action, " position1: ", position1, " position2: ", position2)
                        return board, 0
    return board, 1



"""Handles the creation of a board. Performs turn-order checking
    and calls other functions for actions to be performed"""
def createBoard(moves):
    #Make sure correct color is moving.
    #Call movePiece or capturePiece.
    #Check if the piece ends at the end of the board, if so make it king.
    # 0 = empty
    # 1 = black checker
    # 2 = black king
    # 3 = white checker
    # 4 = white king
    #starting_board = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    board = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    #whose_turn: 0 = black, 1 = white
    whose_turn = 0
    errorcheck = 1


    if not(isinstance(moves, list)):
        # Does not have to be three, just meant to be a plain check for whether the moves are a valid set of moves.
        # Some move-data is a single integer, which is not enough to describe a board/match
        return board, 0

    for move in moves:
        checker_to_move = str(move[0])
        if(move[1].isdigit()):
            checker_to_move = checker_to_move + str(move[1])
        checker_to_move = int(checker_to_move)
        if(whose_turn == 0):
            if(board[checker_to_move -1] == 1 or board[checker_to_move -1] == 2):
                    #turn order ok
                    if (move.__contains__("-")):
                        board, errorecheck = movePiece(board, move, checker_to_move)
                    elif(move.__contains__("x")):
                        board, errorcheck = capturePiece(board, move, checker_to_move)
                    else:
                        print("Error 4, move does not contain - nor x")
                        return board, 0
                    if(errorcheck == 0):
                        return board, 0
                    whose_turn = 1

            else:
                print("Error 5, wrong turn order, Black's turn now, not White. Or trying to move non-existing piece: ", board[checker_to_move -1], " on position: ", checker_to_move)
                print("Moves: ", moves)
                print("Move: ", move)
                print(board)
                return board, 0
        elif(whose_turn == 1):
            if (board[checker_to_move - 1] == 3 or board[checker_to_move - 1] == 4):
                #turn order ok
                if (move.__contains__("-")):
                    board, errorcheck = movePiece(board, move, checker_to_move)
                elif (move.__contains__("x")):
                    board, errorcheck = capturePiece(board, move, checker_to_move)
                else:
                    print("Error 4, move does not contain - nor x")
                    return board, 0
                if(errorcheck == 0):
                    return board, 0
                whose_turn = 0
            else:
                print("Error 6, wrong turn order, White's turn now, not Black. Or trying to move non-existing piece: ", board[checker_to_move -1], " on position: ", checker_to_move)
                print("Moves: ", moves)
                print("Move: ", move)
                print(board)
                return board, 0

        else:
            print("Error 7, turn error in code")
            return board, 0

    return board, 1


print("Creating boards")
boards = []
for i in range(len(extracted_data)):
    this_board = []
    created_board, errorcheck = createBoard(extracted_data[i][0])
    if(errorcheck == 1):
        this_board.append(created_board)
        this_board.append(extracted_data[i][1])
        boards.append(this_board)
    else:
        bad_boards += 1
        print("Bad board, total bad boards: ", bad_boards)


#print(boards)
#print(bad_boards)


def round_down(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier) / multiplier


"""Transforms data into metadata:

1. Number of own uncrowned pieces
2. Number of opponent uncrowned pieces
3. Number of own kings
4. Number of opponent kings
5. Number of own pieces (crowned and uncrowned) on left and right edges of board
6. Integer value of own vertical center of mass
7. Integer value of opponent vertical center of mass


Notes: 
Number of own uncrowned pieces
Number of opponent uncrowned pieces
Number of own kings
Number of opponent kings
Number of own pieces (crowned and uncrowned) on left and right edges of board
	counts only places completely on the left and right sides.
		3,4,11,12,19,20,27,28
Integer value of own vertical center of mass
	I think I will simply let the value represent the row where the vertical center of mass is: 0-7.
    Kings are worth twice as much as pawns.
Integer value of opponent vertical center of mass
	I think I will simply let the value represent the row where the vertical center of mass is: 0-7.
	Kings are worth twice as much as pawns.
	    	0-3
            4-7
            8-11
            12-15
            16-19
            20-23
            24-27
            28-30"""

def metaDataized(boards):
    metadatas = []
    #Iterates through the list of boards
    for i in range(len(boards)):
        this_board = []
        black_pawns = 0
        black_kings = 0
        white_pawns = 0
        white_kings = 0
        own_pieces_on_sides = 0

        black_vertical_weights = []
        for r in range(8):
            black_vertical_weights.append(0)

        white_vertical_weights = []
        for r in range(8):
            white_vertical_weights.append(0)
        #Iterates through each board
        for e in range(len(boards[i][0])):
            if(boards[i][0][e] == 1):
                black_pawns += 1
            elif(boards[i][0][e] == 2):
                black_kings += 1
            elif(boards[i][0][e] == 3):
                white_pawns += 1
            elif(boards[i][0][e] == 4):
                white_kings += 1

            #Full sides used, not counting the spots that are the most to the side that is playable unless there is no unplayable nor playable spot more to the side.
            #3, 4, 11, 12, 19, 20, 27, 28 (+1 for board picture, this is done as lists starts at 0.
            if((e == 3 or e == 4 or e == 11 or e == 12 or e == 19 or e == 20 or e == 27 or e == 28) and (boards[i][0][e] == 1 or boards[i][0][e] == 2)):
                own_pieces_on_sides += 1

            #Getting weights
            if((boards[i][0][e] == 1 or boards[i][0][e] == 2) and (e in list([0,1,2,3]))):
                if(boards[i][0][e] == 1):
                    black_vertical_weights[0] += 1
                elif(boards[i][0][e] == 2):
                    black_vertical_weights[0] += 2
            elif((boards[i][0][e] == 1 or boards[i][0][e] == 2) and (e in list([4,5,6,7]))):
                if (boards[i][0][e] == 1):
                    black_vertical_weights[1] += 1
                elif (boards[i][0][e] == 2):
                    black_vertical_weights[1] += 2
            elif((boards[i][0][e] == 1 or boards[i][0][e] == 2) and (e in list([8,9,10,11]))):
                if (boards[i][0][e] == 1):
                    black_vertical_weights[2] += 1
                elif (boards[i][0][e] == 2):
                    black_vertical_weights[2] += 2
            elif ((boards[i][0][e] == 1 or boards[i][0][e] == 2) and (e in list([12,13,14,15]))):
                if (boards[i][0][e] == 1):
                    black_vertical_weights[3] += 1
                elif (boards[i][0][e] == 2):
                    black_vertical_weights[3] += 2
            elif ((boards[i][0][e] == 1 or boards[i][0][e] == 2) and (e in list([16,17,18,19]))):
                if (boards[i][0][e] == 1):
                    black_vertical_weights[4] += 1
                elif (boards[i][0][e] == 2):
                    black_vertical_weights[4] += 2
            elif ((boards[i][0][e] == 1 or boards[i][0][e] == 2) and (e in list([20,21,22,23]))):
                if (boards[i][0][e] == 1):
                    black_vertical_weights[5] += 1
                elif (boards[i][0][e] == 2):
                    black_vertical_weights[5] += 2
            elif ((boards[i][0][e] == 1 or boards[i][0][e] == 2) and (e in list([24,25,26,27]))):
                if (boards[i][0][e] == 1):
                    black_vertical_weights[6] += 1
                elif (boards[i][0][e] == 2):
                    black_vertical_weights[6] += 2
            elif ((boards[i][0][e] == 1 or boards[i][0][e] == 2) and (e in list([28,29,30,31]))):
                if (boards[i][0][e] == 1):
                    black_vertical_weights[7] += 1
                elif (boards[i][0][e] == 2):
                    black_vertical_weights[7] += 2


            elif ((boards[i][0][e] == 3 or boards[i][0][e] == 4) and (e in list([0,1,2,3]))):
                if (boards[i][0][e] == 3):
                    white_vertical_weights[0] += 1
                elif (boards[i][0][e] == 4):
                    white_vertical_weights[0] += 2
            elif ((boards[i][0][e] == 3 or boards[i][0][e] == 4) and (e in list([4,5,6,7]))):
                if (boards[i][0][e] == 3):
                    white_vertical_weights[1] += 1
                elif (boards[i][0][e] == 4):
                    white_vertical_weights[1] += 2
            elif ((boards[i][0][e] == 3 or boards[i][0][e] == 4) and (e in list([8,9,10,11]))):
                if (boards[i][0][e] == 3):
                    white_vertical_weights[2] += 1
                elif (boards[i][0][e] == 4):
                    white_vertical_weights[2] += 2
            elif ((boards[i][0][e] == 3 or boards[i][0][e] == 4) and (e in list([12,13,14,15]))):
                if (boards[i][0][e] == 3):
                    white_vertical_weights[3] += 1
                elif (boards[i][0][e] == 4):
                    white_vertical_weights[3] += 2
            elif ((boards[i][0][e] == 3 or boards[i][0][e] == 4) and (e in list([16,17,18,19]))):
                if (boards[i][0][e] == 3):
                    white_vertical_weights[4] += 1
                elif (boards[i][0][e] == 4):
                    white_vertical_weights[4] += 2
            elif ((boards[i][0][e] == 3 or boards[i][0][e] == 4) and (e in list([20,21,22,23]))):
                if (boards[i][0][e] == 3):
                    white_vertical_weights[5] += 1
                elif (boards[i][0][e] == 4):
                    white_vertical_weights[5] += 2
            elif ((boards[i][0][e] == 3 or boards[i][0][e] == 4) and (e in list([24,25,26,27]))):
                if (boards[i][0][e] == 3):
                    white_vertical_weights[6] += 1
                elif (boards[i][0][e] == 4):
                    white_vertical_weights[6] += 2
            elif ((boards[i][0][e] == 3 or boards[i][0][e] == 4) and (e in list([28,29,30,31]))):
                if (boards[i][0][e] == 3):
                    white_vertical_weights[7] += 1
                elif (boards[i][0][e] == 4):
                    white_vertical_weights[7] += 2


        #To find center of mass:
        # In this scenario, there is a list of all vertical positions in the board.
        # In order to find the center of mass, each value in the list is multiplied by the number
        # it is in the list; 1-8.
        # These are then summed up and divided by the sum of all values without multiplication.
        # The result is then rounded and we get the center of mass.
        # https://www.softschools.com/formulas/physics/center_of_mass_formula/87/

        black_vertical_multiplied = 0
        black_vertical_summed = 0

        for q in range(len(black_vertical_weights)):
            black_vertical_multiplied += black_vertical_weights[q] * (q+1)
            black_vertical_summed += black_vertical_weights[q]
        black_vertical_center = round_down(black_vertical_multiplied / black_vertical_summed)


        white_vertical_multiplied = 0
        white_vertical_summed = 0

        for q in range(len(white_vertical_weights)):
            white_vertical_multiplied += white_vertical_weights[q] * (q + 1)
            white_vertical_summed += white_vertical_weights[q]
        white_vertical_center = round_down(white_vertical_multiplied / white_vertical_summed)


        this_board.append(black_pawns)
        this_board.append(white_pawns)
        this_board.append(black_kings)
        this_board.append(white_kings)
        this_board.append(own_pieces_on_sides)
        this_board.append(black_vertical_center)
        this_board.append(white_vertical_center)

        to_append = []
        to_append.append(this_board)
        to_append.append(boards[i][1])
        metadatas.append(to_append)
    return metadatas


boards = metaDataized(boards)

def intToBinary(number):
    number = int(number)
    binary_string = '{0:04b}'.format(number)
    binary = []
    for letter in binary_string:
        binary.append(letter)
    return binary

"""Transform into pure binary"""
print("Transforming boards to binary")
binary_boards = []
for i in range(len(boards)):
    if(len(boards[i][0]) != 7):
        print("Error in length of board, not equal to 7")
    this_binary_board = []
    for q in range(len(boards[i][0])):
        in_byte = intToBinary(boards[i][0][q])
        for w in range(len(in_byte)):
            this_binary_board.append(in_byte[w])

    this_binary_board.append(boards[i][1])
    binary_boards.append(this_binary_board)

#print(binary_boards[3])


print("Checking for duplicates")

def removeDuplicates(raw_list):
    no_duplicate_list = []
    loss = 0
    win = 0
    draw = 0
    for q in range(len(raw_list)):
        unique = 1
        for w in range(len(raw_list)):
            if(q != w and raw_list[q] == raw_list[w]):
                unique = 0
        if(unique == 1):
            no_duplicate_list.append(raw_list[q])
            if(raw_list[q][-1] == 0):
                loss += 1
            elif (raw_list[q][-1] == 1):
                win += 1
            elif (raw_list[q][-1] == 2):
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
    return no_duplicate_list, loss, win, draw



original_length = len(binary_boards)
binary_boards, losses, wins, draws = removeDuplicates(binary_boards)
length_without_duplicates = len(binary_boards)
print("Amount of duplicates: ", original_length-length_without_duplicates)



print("Information:")
print("Wins: ", wins)
print("Losses: ", losses)
print("Draws: ", draws)
print("Bad boards: ", bad_boards)
print("Good boards: ", len(binary_boards))



filename = "binaryboards.csv"
print("Storing data as, ", filename)

with open(filename, 'w', newline='') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerows(binary_boards)

