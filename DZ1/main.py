#import numpy
#import matplotlib
#import pandas
#import scipy
import csv


def main():
    file_name = "2016.02.20.vyygrusska.csv"
    p1_fraud = []
    p2_fraud = []
    p3_fraud = []
    read_csv(file_name,p1_fraud,p2_fraud,p3_fraud)


# Read csv file and make tuples of different frauds
def read_csv(file_name, p1_fraud, p2_fraud, p3_fraud):
    reader = csv.reader(open(file_name, "rb"), delimiter=';')
    next(reader)
    for row in reader:
        try:
            p1_fraud.append([row[25], row[30]])
            p2_fraud.append([row[26], row[30]])
            p3_fraud.append([row[27], row[30]])
        except IndexError:
            pass


if __name__ == "__main__":
    main()
