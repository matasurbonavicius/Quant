import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit


# Path to the folder containing the pictures
folder_path = 'path'

# List to store data before creating DataFrame
data = {'id': [], 'size': [], 'width': []}

# Iterate over files in the folder
for filename in os.listdir(folder_path):
    # Check if the file is a regular file (not a directory)
    if os.path.isfile(os.path.join(folder_path, filename)):
        # Split the filename into parts using '-' as the delimiter
        parts = filename.split('-')

        # Extract information from the filename
        if len(parts) == 3:
            image_id, size, width_ext = parts
            # Remove the '.png' extension from the 'width' value
            #width = width_ext.split('.')[0]
            data['id'].append(image_id)
            data['size'].append(size)
            data['width'].append(width_ext)

# Create a DataFrame from the collected data
df = pd.DataFrame(data)

# Set 'id' as the index
df.set_index('id', inplace=True)

# Display the resulting DataFrame
print(df)