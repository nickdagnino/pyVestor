import os
import pandas as pd

file_directory = "Tickers"

data = pd.DataFrame(columns=['symbol', 'description', 'exchange'])

for filename in os.listdir(file_directory):
    if filename.endswith(".txt"):

        filepath = os.path.join(file_directory, filename)

        with open(filepath, 'r') as file:
             for line in file:
                 
                 symbol = line[0:]
                 description = line.strip().split()
                 exchange = "hello"

                 data = data._append({'symbol': symbol, 'description': description, 'exchange': exchange}, ignore_index=True)

data.to_csv("compiled.csv", index = False)

print("DONE")


