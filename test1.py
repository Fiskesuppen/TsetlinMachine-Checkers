



#Compare internally within each list, and between both lists
def removeKFoldDuplicates(train, test):
    no_duplicate_train = []
    no_duplicate_test = []
    trloss = 0
    trwin = 0
    trdraw = 0
    for q in range(len(train)):
        unique = 1
        for w in range(len(train)):
            if(q != w and train[q] == train[w]):
                unique = 0
        if(unique == 1):
            no_duplicate_train.append(train[q])
            if(train[q][-1] == 0):
                trloss += 1
            elif (train[q][-1] == 1):
                trwin += 1
            elif (train[q][-1] == 2):
                trdraw += 1
            else:
                print("ERROR 15, No result to record")
                exit(15)
        else:
            if(not no_duplicate_train.__contains__(train[q])):
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
            if(q != w and test[q] == [test[w]]):
                unique = 0
        if(unique == 1):
            no_duplicate_test.append(test[q])
            if(test[q][-1] == 0):
                teloss += 1
            elif (test[q][-1] == 1):
                tewin += 1
            elif (test[q][-1] == 2):
                tedraw += 1
            else:
                print("ERROR 15, No result to record")
                exit(15)
        else:
            if(not no_duplicate_test.__contains__(test[q])):
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
            if(no_duplicate_test[q] == no_duplicate_train[w]):
                unique = 0
        if(unique == 1):
            cross_no_duplicate_test.append(no_duplicate_test[q])
            if(test[q][-1] == 0):
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



train_list = []
data = [[1,2,3],2]
train_list.append(data)
data = [[2,3,6],2]
train_list.append(data)
data = [[1,2,3],2]
train_list.append(data)
data = [[1,2,3],2]
train_list.append(data)
data = [[1,2,3],2]
train_list.append(data)
data = [[1,245,3],2]
train_list.append(data)
data = [[1,2,39],2]
train_list.append(data)



test_list = []
data = [[1,2,3],2]
test_list.append(data)
data = [[1,2,3],2]
test_list.append(data)
data = [[1,2,39],2]
test_list.append(data)
data = [[1,2,3],2]
test_list.append(data)
data = [[441,2,35],2]
test_list.append(data)

print(train_list)

no_dupe_train, no_dupe_test = removeKFoldDuplicates(train_list, test_list)

print("Train data:")
print(no_dupe_train)
print("Test data:")
print(no_dupe_test)
