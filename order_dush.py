#!/usr/bin/env python3

import os
import getopt


#Bash line to run to get the file sizes. Pipe output to python variable
out=os.popen('du -sh ./*')

#Byte Dict to multiply with
byte_dict = {"P":1e15, "T":1e12, "G":1e9, "M":1e6, "K": 1e3}

#Make list to store lines in
list_of_tups=[]

#Loop through all lines of output from du -sh that were piped to variable
for line in out:
 
    #Split line by white spaces (makes easier to get unit (i.e. P,T,G,M,K)
    line_split=line.split()
    
    #Determine if G,M,K,T
    unit=line_split[0][-1]

    #Get size
    size=float(line_split[0][:-1])
    
    #Actual Size computed using values from byte dict
    size_in_bytes=(size * byte_dict[unit])

    #Add size and line as tuple to list
    list_of_tups.append((size_in_bytes,line))


#Use lambda expression to sort tuples by first entry of tuple
lines_sorted_by_size=sorted(list_of_tups,key=lambda x: x[0],reverse=True)

#print sorted line to screen
for line in lines_sorted_by_size:
    print(line[1].strip("\n"))






