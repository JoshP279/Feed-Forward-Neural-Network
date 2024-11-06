import matplotlib.pyplot as plt
import csv

def load_sse_from_csv(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        sse_values = next(reader)
        return [float(sse) for sse in sse_values]

# Load SSE values from CSV files
sse_values1 = load_sse_from_csv('properWeights.csv')
sse_values2 = load_sse_from_csv('noise.csv')
sse_values3 = load_sse_from_csv('overfitting.csv')
sse_values4 = load_sse_from_csv('all.csv')


sse_values1 = [sse for sse in sse_values1 if sse != 0]
sse_values2 = [sse for sse in sse_values2 if sse != 0]
sse_values3 = [sse for sse in sse_values3 if sse != 0]
sse_values4 = [sse for sse in sse_values4 if sse != 0]

# Create epoch lists for plotting based on the length of each SSE list
epochs1 = list(range(1, len(sse_values1) + 1))
epochs2 = list(range(1, len(sse_values2) + 1))
epochs3 = list(range(1, len(sse_values3) + 1))
epochs4 = list(range(1, len(sse_values4) + 1))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(epochs1, sse_values1, label='Appropriate Weight Initialisation', color='blue', marker='o', linestyle='-', markersize=5)
plt.plot(epochs2, sse_values2, label='Noise Injection', color='green', marker='o', linestyle='-', markersize=5)
plt.plot(epochs3, sse_values3, label='Prevention of Overfitting', color='red', marker='o', linestyle='-', markersize=5)
plt.plot(epochs4, sse_values4, label='All', color='purple', marker='o', linestyle='-', markersize=5)

plt.grid(True, linestyle='--', alpha=0.7)
plt.title('SSE Over Iterations for Different Investigated Techniques')
plt.xlabel('Epochs')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.legend(loc='upper right')
plt.show()
