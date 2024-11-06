import matplotlib.pyplot as plt
import csv

def load_sse_from_csv(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        sse_values = next(reader)
        return [float(sse)/2.0 for sse in sse_values]

# Load SSE values from CSV files
sse_values1 = load_sse_from_csv('sseOverEpochs1.csv')
sse_values2 = load_sse_from_csv('sseOverEpochs2.csv')
sse_values3 = load_sse_from_csv('sseOverEpochs3.csv')
print(sse_values3[-10:])

plt.figure(figsize=(10, 6))

# Plot with enhanced marker styles
plt.plot(sse_values1, label='0.1', color='blue', marker='o', linestyle='-', markersize=5, markevery=50)
plt.plot(sse_values2, label='0.01', color='green', marker='o', linestyle='-', markersize=5, markevery=50)
plt.plot(sse_values3, label='0.001', color='red', marker='o', linestyle='-', markersize=5, markevery=50)

# Add grid, labels, and legend
plt.grid(True, linestyle='--', alpha=0.7)
plt.title('SSE Over Iterations for Different Learning Rates')
plt.xlabel('Epochs')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.legend(loc='upper right')
plt.show()
