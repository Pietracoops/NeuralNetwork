#=============================================
# Script Name: StockHistory.py
# Written by: Massimo Pietracupa
# Date: 2019-08-24
# Version: 1.0
#
# Description: This script is used to retrieve
# the stock history of a stocks and store them
# as .xlsx and .csv.
#
#
# Number of Inputs: 2
#	Input 1: Stock Name (e.g. RY.TO)
#	Input 2: Folder destination of output
#
#=============================================
import pandas_datareader as pdr
import pandas as pd
import sys
import xlrd
import csv
from datetime import datetime
from pandas_datareader.data import get_quote_yahoo


StockName = sys.argv[1]
Directory = sys.argv[2]

#print 'Number of arguments:', len(sys.argv), 'arguments.'
#print 'Argument List:', str(sys.argv)

today = datetime.today()
#datem = datetime(today.year, today.month, 1)

stockinfo = pdr.get_data_yahoo(StockName, start=datetime(1970, 1, 1), end=datetime(today.year, today.month, today.day))

#print(stockinfo['Adj Close'])	# Example of printing a single bit of information
#print(stockinfo)				# Example of printing all info


df = pd.DataFrame(stockinfo).T
df.to_excel(excel_writer = Directory + "/" + StockName + ".xlsx")

print("Generating to csv")

wb = xlrd.open_workbook(StockName + ".xlsx")
sh = wb.sheet_by_name('Sheet1')
your_csv_file = open(StockName + '.csv', 'w')
wr = csv.writer(your_csv_file, quoting=csv.QUOTE_ALL)

for rownum in range(sh.nrows):
    wr.writerow(sh.row_values(rownum))

your_csv_file.close()

