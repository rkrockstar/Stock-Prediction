import csv
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model

txt_file = r"aapl_complete.txt"
csv_file = r"mycsv.csv"
in_txt = csv.reader(open(txt_file, "r"), delimiter = '\t')
out_csv = csv.writer(open(csv_file, 'w', newline=''))

for row in in_txt:
    if any(row):
        out_csv.writerow(row)
