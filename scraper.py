import os
import csv
import requests

# The original link
link = 'your_link_here'

# The folder containing the .csv files
folder_path = 'your_folder_path_here'

# Iterate over the .csv files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        with open(os.path.join(folder_path, filename), 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                # Modify the link using data from the .csv file
                link = link + 'your_modification_here'

# Fetch the file from the modified link
response = requests.get(link, allow_redirects=True)

# The path to the specific folder where the file should be saved
save_path = 'your_specific_folder_here'

# Save the file
with open(os.path.join(save_path, 'downloaded_file'), 'wb') as f:
    f.write(response.content)
