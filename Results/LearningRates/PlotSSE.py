import pandas as pd
import matplotlib.pyplot as plt

# Replace 'sse_data.csv' with your actual file path
file_path = 'sseOverEpochs.csv'

# Read the CSV file with no headers
data = pd.read_csv(file_path, header=None)

# Assuming the data is in the first row and multiple columns
sse_values = data.iloc[0]

# Plotting the SSE values
plt.figure(figsize=(8, 5))
plt.plot(sse_values, marker='o')
plt.title('Sum of Squared Errors (SSE)')
plt.xlabel('Index')
plt.ylabel('SSE')
plt.grid(True)
plt.show()