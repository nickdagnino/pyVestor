import datetime as date
import os
import csv
import requests

directory = 'Tickers'

period1 = 0
period2 = date.datetime.now().timestamp()
interval = ['1d', '1wk', '1mo']
event = ['history', 'div', 'split', 'capitalGain']
adjustedClose = ['true', 'false']

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

link = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval={interval}&events={event}&includeAdjustedClose={adjustedClose}"

for file in os.listdir(directory):
    
    filepath = os.path.join(directory, file)
    print(f'Reading file: {filepath}')

    



#ticker = "NVDA"
#link = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval={interval}&events={event}&includeAdjustedClose={adjustedClose}"

#response = requests.get(link, headers=headers)

#assert response.status_code == 200

#folder_path = 'Market Data\\TLSA.csv'

#with open(folder_path, 'wb') as file:
#    file.write(response.content)
