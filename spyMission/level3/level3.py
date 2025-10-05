import csv

with open('train_data.csv', 'r') as inFile, open('updated_train_data.csv', 'w') as outFile:
    reader = csv.reader(inFile)
    writer = csv.writer(outFile)

    header = next(reader)
    writer.writerow(header)

    for row in reader:
        isGood = True
        if float(row[2]) < 0 or float(row[2]) > 1:
            isGood = False
        if float(row[3]) < 0.02 or float(row[3]) > 8.7:
            isGood = False
        if float(row[4]) < -100 or float(row[4]) > 373.15:
            isGood = False
        if float(row[6]) < -1 or float(row[6]) > 1:
            isGood = False
        if float(row[7]) < -2 or float(row[7]) > 2:
            isGood = False
        if row[8] == 'NA':
            isGood = False
        if isGood:
            writer.writerow(row)
