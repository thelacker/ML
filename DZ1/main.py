import matplotlib.pyplot as plot
import numpy as np
from sklearn.metrics import roc_curve, auc
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

    # finding the stats
    p1_fraud_stats = stats(p1_fraud, 0.5)
    p2_fraud_stats = stats(p2_fraud, 0.5)
    p3_fraud_stats = stats(p3_fraud, 0.5)

    # showing the stats of each Magic Box
    print "p1_fraud stats: " + str(p1_fraud_stats[0]) + "\nThreshold: " + str(find_threshold(p1_fraud))
    print "\np2_fraud stats: " + str(p2_fraud_stats[0]) + "\nThreshold: " + str(find_threshold(p2_fraud))
    print "\np3_fraud stats: " + str(p3_fraud_stats[0]) + "\nThreshold: " + str(find_threshold(p3_fraud))

    # making curves
    build_roc_curve(p1_fraud_stats[1], p1_fraud_stats[2])
    build_roc_curve(p2_fraud_stats[1], p2_fraud_stats[2])
    build_roc_curve(p3_fraud_stats[1], p3_fraud_stats[2])

# Counting tp, pn, fp, fn
def stats(cases, threshold):
    all_cases = fraud_cases = granted_cases = tp_cases = tn_cases = fp_cases = fn_cases = 0
    actual = list()
    predictions = list()
    for case in cases:
        # if the probability is bigger then threshold
        # and case was Fraud - then the Magic Box is right
        if case[1] == 'F':
            fraud_cases += 1

            # to build an roc-curve, we have to build actual and prediction lists
            actual.append(1)
            predictions.append(float(case[0]))

            if float(case[0]) >= threshold:
                tp_cases += 1

        # if the probability is bigger then threshold
        # and case was Granted - then the Magic Box is wrong
        elif case[1] == "G":
            granted_cases += 1

            # to build an roc-curve, we have to build actual and prediction lists
            actual.append(0)
            predictions.append(float(case[0]))

            if float(case[0]) >= threshold:
                fp_cases += 1
        # If the result is undefined - skip case
        else:
            continue

    # counting other data for stats
    all_cases = fraud_cases + granted_cases
    fn_cases = fraud_cases - tp_cases
    tn_cases = granted_cases - fp_cases

    # counting the the stats and returning them in a dictionary
    statistic = {'tp': float(tp_cases)/fraud_cases, 'fp': float(fn_cases)/granted_cases,
                 'tn': float(tn_cases)/granted_cases, 'fn': float(fn_cases)/fraud_cases}
    return [statistic, actual, predictions]


def build_roc_curve(actual, predictions):
    # building a roc curve
    plot.figure()
    fp_rate, tp_rate, threshold = roc_curve(actual, predictions)
    roc_auc = auc(fp_rate, tp_rate)

    # adjusting the plot
    plot.plot(fp_rate, tp_rate, label='%s ROC, AUC = %0.2f, Gini = %0.2f' % ('p_fraud', roc_auc, (roc_auc * 2) - 1))
    plot.title('ROC Curve')
    plot.legend(loc='lower right', fontsize='small')
    plot.plot([0,1],[0,1],'r--')
    plot.xlim([0.0,1.0])
    plot.ylim([0.0,1.0])
    plot.ylabel('TP Rate')
    plot.xlabel('FP Rate')
    plot.show()

# Function to find the threshold for every Magic Box to make fp not more than 0.2
def find_threshold(system):
    threshold = 0.9
    while True:
        fp = stats(system, threshold)[0]['fp']
        if fp > 0.2:
            threshold -= 0.0009
        else:
            return threshold


if __name__ == "__main__":
    main()
