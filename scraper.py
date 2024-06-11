import datetime as date
import os
import csv
import requests

period1 = 0
period2 = date.datetime.now().timestamp()
interval = ['1d', '1wk', '1mo']
event = ['history', 'div', 'split', 'capitalGain']
adjustedClose = ['true', 'false']

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

ticker = "NVDA"
link = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval={interval}&events={event}&includeAdjustedClose={adjustedClose}"

response = requests.get(link, headers=headers)

assert response.status_code == 200

folder_path = 'Market Data\\TLSA.csv'

with open(folder_path, 'wb') as file:
    file.write(response.content)

#webbrowser.open(link)

# The folder containing the .csv files
#folder_path = 'Market Data'

# Iterate over the .csv files in the folder
#for filename in os.listdir(folder_path):
#    if filename.endswith('.csv'):
#        with open(os.path.join(folder_path, filename), 'r') as f:
#            reader = csv.reader(f)
#            for row in reader:
#                # Modify the link using data from the .csv file
#                link = link + 'your_modification_here'

# Fetch the file from the modified link
#response = requests.get(link, allow_redirects=True)
#print(response)

# The path to the specific folder where the file should be saved
#save_path = 'Market Data'

# Save the file
#with open(os.path.join(save_path, 'downloaded_file'), 'wb') as f:
#    f.write(response.content)
