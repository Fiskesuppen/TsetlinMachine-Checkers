


#somestring = "1. 11-15 24-20 2. 3x19x2 28-24 3. 10x3x22x44x51 22-18 4. 15x22 25x18 5. 4-8 26-22 6. 10-14"
local_playstring = "10x3x22x44x51 24-20 2. 3x19x2 28-24 3. 10x3x22x44x51 22-18 4. 15x22 25x18 5. 4-8 26-22 6. 10-14 10x3x22x44x51"

somelist = []
capturetimeout = 0

for q in range(len(local_playstring)):
    thisaction = ""
    if(local_playstring[q] == "-"):
        if(q > 0):
            if(local_playstring[q - 1].isdigit()):
                if(q > 1):
                    if(local_playstring[q - 2].isdigit()):
                        thisaction = thisaction + local_playstring[q - 2]
                thisaction = thisaction + local_playstring[q - 1]
                thisaction = thisaction + local_playstring[q]
        if(q < len(local_playstring) -1):
            if(local_playstring[q + 1].isdigit()):
                thisaction = thisaction + local_playstring[q + 1]
                if(q < len(local_playstring) -2):
                    if(local_playstring[q + 2].isdigit()):
                        thisaction = thisaction + local_playstring[q + 2]
                somelist.append(thisaction)
    elif(local_playstring[q] == "x"):
        if(capturetimeout == 0):
            capturetimeout = 1
            if(q > 0):
                if(local_playstring[q - 1].isdigit()):
                    if(q > 1):
                        if (local_playstring[q - 2].isdigit()):
                            thisaction = thisaction + local_playstring[q - 2]
                    thisaction = thisaction + local_playstring[q - 1]
                    thisaction = thisaction + local_playstring[q]
            w = 1
            while(w < len(local_playstring)):
                iter = q + w
                w += 1
                if(iter < len(local_playstring)):
                    if(local_playstring[iter].isdigit()):
                        thisaction = thisaction + local_playstring[iter]
                    elif(local_playstring[iter] == "x"):
                        if(local_playstring[iter - 1] == "x"):
                            print("ERROR, double x at index: ", iter-1)
                        else:
                            thisaction = thisaction + local_playstring[iter]
                    else:
                        w = len(local_playstring) + 20
                else:
                    w = len(local_playstring) + 20
            somelist.append(thisaction)
    elif(local_playstring[q] == " "):
        capturetimeout = 0



print(somelist)