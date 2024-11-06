import matplotlib.pyplot as plt
import csv

def load_sse_from_csv(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        sse_values = next(reader)
        return [float(sse) for sse in sse_values]

# Load SSE values from CSV files
sse_values1 = load_sse_from_csv('sseOverEpochs1.csv')
sse_values2 = load_sse_from_csv('sseOverEpochs2.csv')
sse_values3 = load_sse_from_csv('sseOverEpochs3.csv')

# Determine the maximum length among all SSE lists
max_length = max(len(sse_values1), len(sse_values2), len(sse_values3))

# Extend SSE lists to match the maximum length by repeating the last value
def extend_to_length(sse_values, length):
    return sse_values + [sse_values[-1]] * (length - len(sse_values))

sse_values1 = extend_to_length(sse_values1, max_length)
sse_values2 = extend_to_length(sse_values2, max_length)
sse_values3 = extend_to_length(sse_values3, max_length)

# Create epoch lists for plotting
epochs = list(range(1, max_length + 1))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(epochs, sse_values1, label='0.1', color='blue', marker='o', linestyle='-', markersize=5)
plt.plot(epochs, sse_values2, label='0.01', color='green', marker='s', linestyle='-', markersize=5)
plt.plot(epochs, sse_values3, label='0.001', color='red', marker='^', linestyle='-', markersize=5)

plt.grid(True, linestyle='--', alpha=0.7)
plt.title('SSE Over Epochs for Different Learning Rates')
plt.xlabel('Epochs')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.legend(loc='upper right')
plt.show()
