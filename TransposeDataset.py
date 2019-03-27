import csv
#from itertools import zip

a = zip(*csv.reader(open("D:\\Thesis\\Implementation\\Main DataSet\\brca_CNA_data_v01.csv", "rt")))
csv.writer(open("D:\\Thesis\\Implementation\\Main DataSet\\A2_BRCAA.csv", "wt")).writerows(a)
