import numpy
import matplotlib
import pandas
import scipy
import csv


def main():
    # File with data
    file_name = "2016.02.20.vyygrusska.csv"

    # Different Magic Boxes
    p1_fraud = []
    p2_fraud = []
    p3_fraud = []

    read_csv(file_name,p1_fraud,p2_fraud,p3_fraud)


# Read csv file and make tuples of different frauds
def read_csv(file_name, p1_fraud, p2_fraud, p3_fraud):
    # Reading file and kipping the header
    reader = csv.reader(open(file_name, "rb"), delimiter=';')
    next(reader)

    for row in reader:
        try:
            p1_fraud.append([row[25], row[30]])
            p2_fraud.append([row[26], row[30]])
            p3_fraud.append([row[27], row[30]])
        except IndexError:
            # There are some broken rows, we have to skip them
            pass

    # showing the stats of each Magic Box
    print "p1_fraud stats: " + str(stats(p1_fraud, 0.5)) + "\nThreshold: " + str(find_threshold(p1_fraud))
    print "\np2_fraud stats: " + str(stats(p2_fraud, 0.5)) + "\nThreshold: " + str(find_threshold(p2_fraud))
    print "\np3_fraud stats: " + str(stats(p3_fraud, 0.5)) + "\nThreshold: " + str(find_threshold(p3_fraud))

# Counting tp, pn, fp, fn
def stats(cases, threshold):
    all_cases = frod_cases = granted_cases = tp_cases = tn_cases = fp_cases = fn_cases = 0
    for case in cases:
        # if the probability is bigger then threshold
        # and case was Frod - then the Magic Box is right
        if case[1] == 'F':
            frod_cases += 1
            if float(case[0]) >= threshold:
                tp_cases += 1
        # if the probability is bigger then threshold
        # and case was Granted - then the Magic Box is wrong
        elif case[1] == "G":
            granted_cases += 1
            if float(case[0]) >= threshold:
                fp_cases += 1
        # If the result is undefined - skip case
        else:
            continue

    # counting other data for stats
    all_cases = frod_cases + granted_cases
    fn_cases = frod_cases - tp_cases
    tn_cases = granted_cases - fp_cases

    # counting the the stats and returning them in a dictionary
    statistic = {'tp': float(tp_cases)/frod_cases, 'fp': float(fn_cases)/granted_cases,
                 'tn': float(tn_cases)/granted_cases, 'fn': float(fn_cases)/frod_cases}
    return statistic


# Function to find the threshold for every Magic Box to make fp not more than 0.2
def find_threshold(system):
    threshold = 0.9
    while True:
        fp = stats(system, threshold)['fp']
        if fp > 0.2:
            threshold -= 0.0009
        else:
            return threshold


if __name__ == "__main__":
    main()
