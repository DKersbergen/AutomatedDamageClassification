import csv

listofEx = []
with open('NN_Exclude_Middle.csv') as csvFile:
    for line in csvFile:
        for i in line.split():
            listofEx.append(int(i))
#print(listofEx)

listI = [11,2,5,1,2,2,2,526820892, 526820893, 521478070, 526820894, 521629433, 526820896, 521478072, 521629435, 521478073, 526820897, 521629437, 521630274, 526820898, 521478074, 521630275, 526820899, 526820900, 521630276, 524728251, 521630277, 526820901, 524728254, 526820902, 521630279, 524728259, 526820903, 521630280, 521630283, 526820904, 521630284, 526820905, 524728533, 521630285, 526820906, 524728536, 526820907, 521630286, 524728537, 526820908, 521630287, 526820909, 521630288, 526820910, 521630289,
521630822, 526820911, 521630823, 526820912, 521630824, 526820913, 521630825, 526820914, 521630826, 526820915, 521630827, 526820916, 521630828, 526820917, 526820920, 521630829, 526820921, 521630830, 526820922, 521630831, 526820923, 521630832, 526820924, 521630833, 526820925, 521630834, 526820926, 521630835, 526820927, 521630836, 526820928, 521630837, 526820929, 521630838, 526820930, 521630839, 526820931, 521630840, 526820932, 521630841, 526820933, 521630846, 526820934, 521630847, 526820935, 526820936, 526820937, 526820938, 526820939, 526820940, 526820941, 526820942, 526820943, 526820944, 526820945, 526820946, 526820947, 526820948, 526820949, 526820950, 526820951, 526820952, 526820953, 526820954, 526820956, 526820957, 521631575, 526820958, 521631790, 521394033, 526820959, 521631791, 526820960, 521631792, 526821535, 521631793, 521394036, 526821536, 521631794, 526821537, 521631795, 521631796, 521631798, 521631799, 521631801, 521631803, 521631804, 521631805, 521394045, 521631806, 521394049, 521631807, 521394056, 521631808, 521394057, 521631809, 521394059, 521631810, 521394060, 521631811, 521394063, 521631812, 521394064, 521631813, 521394065, 521631814, 521394066, 521631816, 521394070, 521631817, 521394071, 521631819, 521394073, 521394074, 521394075, 524731375, 524731376, 524731377, 524731378, 524731379, 525320095, 525325202, 525325203, 525325204, 525325205, 525325206, 525325207, 525325208, 525325209, 525661040, 525325210, 525661041, 525325211, 525661042, 525325212, 525661043, 525325213, 525661044, 525325214, 525661045, 525661046, 525661047, 525661048, 525661049, 525661050, 525661051, 2,2,2,2,2,2,2,5,5,4,2,56,87,4,64,8,16,8,463,1,86,5,158,31,81,35,18,1,851,51,51,5,1,5,15,1,51,5,1,51,5,15,1,51,5,1,51,51,5,1,51,5,15,1,51,51,5,15,1,51,51,5,15,1,51,51,5,1,51,51,5,1,5,524722492]

for j in listI:
    if j in listofEx:
        print(j, "yes")