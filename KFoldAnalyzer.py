import csv

train_lengths = []
train_losses = []
train_wins = []
train_draws = []

test_lengths = []
test_losses = []
test_wins = []
test_draws = []

traintest_ratios = []

basepath = "JupyterLab/NoDupeCheck/StandardPureMetaData/"
basename = "statickfold.data"
for i in range(10):
   train_lengths.append(0)
   train_losses.append(0)
   train_wins.append(0)
   train_draws.append(0)
   test_lengths.append(0)
   test_losses.append(0)
   test_wins.append(0)
   test_draws.append(0)
   filepath = basepath + str(i) + "train" + basename
   with open(filepath) as f:
      reader = csv.reader(f)
      for row in reader:

         train_lengths[i] += 1
         if(row[-1] == "0"):
            train_losses[i] += 1
         elif(row[-1] == "1"):
            train_wins[i] += 1
         elif(row[-1] == "2"):
            train_draws[i] += 1
         else:
            print("No result to rekord in train number ", i, " Line: ", row)
            print(row[-1])
   f.close()

   filepath = basepath + str(i) + "test" + basename
   with open(filepath) as f:
      reader = csv.reader(f)
      for row in reader:
         test_lengths[i] += 1
         if(row[-1] == "0"):
            test_losses[i] += 1
         elif(row[-1] == "1"):
            test_wins[i] += 1
         elif(row[-1] == "2"):
            test_draws[i] += 1
         else:
            print("No result to rekord in test number ", i, " Line: ", row)
   f.close()

   traintest_ratios.append((test_lengths[i]/train_lengths[i])*100)

   print("KFold ", i,
         " Train test ratios:  Size ratio: ", traintest_ratios[i],
         " Loss ratio: ", (test_losses[i]/train_losses[i])*100,
         " Win ratio: ", (test_wins[i]/train_wins[i])*100,
         " Draw ratio: ", (test_draws[i]/train_draws[i])*100)
   print("KFold ", i, " Train: Amount of data: ", train_lengths[i], " Loss percent: ",
         (train_losses[i]/(train_wins[i]+train_draws[i]+train_losses[i]))*100, " Win percent: ",
         (train_wins[i]/(train_losses[i]+train_draws[i]+train_wins[i]))*100, " Draw percent: ",
         (train_draws[i]/(train_losses[i]+train_wins[i]+train_draws[i]))*100)
   print("KFold ", i, " Test: Amount of data: ", test_lengths[i], " Loss percent: ",
         (test_losses[i]/(test_wins[i]+test_draws[i]+test_losses[i]))*100, " Win percent: ",
         (test_wins[i]/(test_losses[i]+test_draws[i]+test_wins[i])*100), " Draw percent: ",
         (test_draws[i]/(test_losses[i]+test_wins[i]+test_draws[i]))*100)


print("Max train/test split variance: ", max(traintest_ratios) - min(traintest_ratios),
      " Largest test ratio: ", max(traintest_ratios), " Smallest test ratio: ", min(traintest_ratios))


#print("Total data: ", totaldata, " Average train/test ratio: ", avgtraintest, " Loss percent: ", losspercent, " Win percent: ", winpercent, " Draw percent: ", drawpercent)
"""
print(train_lengths)
print(train_losses)
print(train_wins)
print(train_draws)

print(test_lengths)
print(test_losses)
print(test_wins)
print(test_draws)
"""

print("")

def Average(lst):
    return sum(lst) / len(lst)

totaldata = train_lengths[0] + test_lengths[0]
print("Total data: ", totaldata)
print("Average train/test ratio: ", Average(traintest_ratios))

losses = sum(train_losses) + sum(test_losses)
wins = sum(train_wins) + sum(test_wins)
draws = sum(train_draws) + sum(test_draws)


lossratio = losses/totaldata
winratio = wins/totaldata
drawratio = draws/totaldata
print("Loss ratio: ", round(lossratio, 2) *10)
#print("Win ratio: ", round(winratio, 2) *10)
print("Win ratio: ", round(winratio, 2) *10)
print("Draw ratio: ", round(drawratio, 2) *10)

print("")
print("Resultratio check: ", (round(lossratio, 2) + round(winratio, 2) + round(drawratio, 2)) *10 )



#for i in range(len(train_lengths)):
    #print(train_lengths[i] + test_lengths[i])
    #print(train_losses[i] + test_losses[i])



