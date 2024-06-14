import datetime as date
import os
#import csv
import requests
import time

directory = 'Tickers'

period1 = -2147483648
period2 = int(date.datetime.now().timestamp())
interval = ['1d', '1wk', '1mo']
event = ['history', 'div', 'split', 'capitalGain']
adjustedClose = ['true', 'false']

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

print('SCRAPE HAS BEGUN')

for file in os.listdir(directory):
    
    filepath = os.path.join(directory, file)
    print(f'Reading file: {filepath}')

    with open(filepath, 'r') as f:
        symbols = f.read().strip().split('\n')
        symbols.remove(symbols[0])

    for symbol in symbols:

        #for info in range(len(symbols)):
            ticker = symbol[:symbol.index('\t')]
            #ticker = symbols[info][:symbols[info].index('\t')]

            if '.' in ticker:
                replace = ticker.split('.')
                ticker = replace[0] + '-' + replace[1]

            print(ticker)

            link = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval={interval[0]}&events={event[0]}&includeAdjustedClose={adjustedClose[0]}'
            
            response = requests.get(link, headers=headers)

            try:
                assert response.status_code == 200
            except AssertionError:
                print(f'Ticker, {ticker}, is invalid.')
        

            folder_path = f'C:\\Users\\Nick Dagnino\\OneDrive\\Desktop\\Market Data\\{ticker}.csv'

            with open(folder_path, 'wb') as file:
                file.write(response.content)

print('SCRAPE HAS TERMINATED')
