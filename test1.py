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


this_list = [[1,2,3,1],[2,3,4,1],[2,5,7,2],[1,2,3,1],[2,3,4,1],[2,3,4,1],[2,3,4,1]]

no_dupes, losses, wins, draws = removeDuplicates(this_list)

print(no_dupes)
print(losses)
print(wins)
print(draws)