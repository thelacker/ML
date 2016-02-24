#import numpy
#import matplotlib
#import pandas
#import scipy
import csv

file_name = "2016.02.20.vyygrusska.csv"

def main():
    reader = csv.reader(open(file_name, "rb"), delimiter=';')
    for row in reader:
        print row[25:28]
        print row[30]

if __name__ == "__main__":
    main()
