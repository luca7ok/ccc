import csv
import math
from itertools import count

with open('train_data.csv', 'r') as inFile, open('updated_train_data.csv', 'w') as outFile:
    reader = csv.reader(inFile)
    writer = csv.writer(outFile)

    header = next(reader)
    writer.writerow(header)

    goodAverage = -0.69
    goodStandDev = 48

    standard_dev = 0
    count_k = 0
    for row in reader:
        isGood = True
        if row[5] == 'K':
            #average += (float(row[4]) - 273.15)
            count_k += 1
            standard_dev += pow(((float(row[4]) - 273.15) - average), 2)
            row[4] = "%.2f" % (float(row[4]) - 273.15)
            row[5] = 'C'
        elif row[5] == '?':
            isGood = False
        if isGood:
            writer.writerow(row)
