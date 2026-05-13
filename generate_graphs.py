import matplotlib.pyplot as plt

RESULT_PATH = "results/poi_grad.txt"
GRAPH_PATH = "results/poi_loss.pdf"

with open(RESULT_PATH, 'r') as file:
        lines = file.readlines()
        # collecting loss
        grad_loss = list(map(float, lines[1].split()[1:]))

# Plot graph
plt.figure(figsize=(7.2, 5.6))
plt.plot(range(len(grad_loss)), grad_loss, color='blue', linestyle='-', label="GD")

plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.title('Loss curve of Poisson Regression')

# Save plot as PDF
plt.savefig(GRAPH_PATH, format='pdf', bbox_inches='tight')
