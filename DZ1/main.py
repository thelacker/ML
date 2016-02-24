#import numpy
#import matplotlib
#import pandas
#import scipy
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

    stats(p1_fraud, 0.5)


# Counting tp, pn, fp, fn
def stats(cases, threshold):
    all_cases = len(cases)
    tp_cases = 0
    tn_cases = 0
    fp_cases = 0
    fn_cases = 0
    for case in cases:
        # If the result is undefined - skip case
        if case[1] == 'U':
            continue

        # if the probability is bigger then threshold
        # and case was Granted - then the Magic Box is right
        if case[0] >= threshold and case[1] == 'F':
            tp_cases += 1
        elif case[0] >= threshold and case[1] == 'G':
            fp_cases += 1
        elif case[0] < threshold and case[1] == 'F':
            fn_cases += 1
        elif case[0] < threshold and case[1] == 'G':
            tn_cases += 1

    print all_cases
    print tp_cases
    print tn_cases
    print fp_cases
    print fn_cases


if __name__ == "__main__":
    main()
