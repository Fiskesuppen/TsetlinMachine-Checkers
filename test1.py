#import numpy as np
#data = np.asarray([ [1,2,3], [4,5,6], [7,8,9] ])

#np.savetxt('test.csv', data, fmt="%d", delimiter=',')

i = 2

somelist = "'26-23 22. 19x26 31x22x15 23. 25-22 14-10 24. 22-18 10x1 25. 18x11 1-6 26. 11-15', '28-24 1/2-1/2'"


checkvar = ""

for z in range(8):
    checkvar = checkvar + somelist[len(somelist)-9+z]


if checkvar.__contains__("0,1"):
    somelist = somelist[:-3]
elif checkvar.__contains__("1,0"):
    somelist = somelist[:-3]
elif checkvar.__contains__("1/2-1/2"):
    somelist = somelist[:-8]
else:
    print("ERROR, no extra result to remove at i number: ", i)

print(checkvar)
print(somelist)

