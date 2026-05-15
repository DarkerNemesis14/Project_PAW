import matplotlib.pyplot as plt

# define paths
DS1_RESULT_PATH = "results/dataset1_results.txt"
DS1_TRAIN_GRAPH_PATH = "results/dataset1_loss.pdf"
DS2_RESULT_PATH = "results/dataset2_results.txt"
DS2_TRAIN_GRAPH_PATH = "results/dataset2_loss.pdf"

# read results for dataset 1
with open(DS1_RESULT_PATH, 'r') as file:
        lines = file.readlines()
        # collecting loss
        grad_train_loss = list(map(float, lines[1].split()[1:]))
        grad_val_loss = list(map(float, lines[2].split()[1:]))
        regu_train_loss = list(map(float, lines[4].split()[1:]))
        regu_val_loss = list(map(float, lines[5].split()[1:]))
        adam_train_loss = list(map(float, lines[7].split()[1:]))
        adam_val_loss = list(map(float, lines[8].split()[1:]))
        adamw_train_loss = list(map(float, lines[10].split()[1:]))
        adamw_val_loss = list(map(float, lines[11].split()[1:]))
        adamwd_train_loss = list(map(float, lines[13].split()[1:]))
        adamwd_val_loss = list(map(float, lines[14].split()[1:]))

# Plot train loss graph
# plt.plot(range(len(grad_train_loss)), grad_train_loss, color='blue', linewidth=2, linestyle='-', label="GD")
plt.plot(range(len(regu_train_loss)), regu_train_loss, color='red', linewidth=2, linestyle='-', label="Regularizer")
plt.plot(range(len(adam_train_loss)), adam_train_loss, color='green', linewidth=2, linestyle='-', label="Adam")
# plt.plot(range(len(adamw_train_loss)), adamw_train_loss, color='orange', linewidth=2, linestyle='-', label="AdamW")
plt.plot(range(len(adamwd_train_loss)), adamwd_train_loss, color='purple', linewidth=2, linestyle='-', label="AdamW with Decay")

# Plot validation loss graph
# plt.plot(range(len(grad_val_loss)), grad_val_loss, color='blue', linewidth=1, linestyle='--', label="GD")
plt.plot(range(len(regu_val_loss)), regu_val_loss, color='red', linewidth=1, linestyle='--')
plt.plot(range(len(adam_val_loss)), adam_val_loss, color='green', linewidth=1, linestyle='--')
# plt.plot(range(len(adamw_val_loss)), adamw_val_loss, color='orange', linewidth=1, linestyle='--', label="AdamW")
plt.plot(range(len(adamwd_val_loss)), adamwd_val_loss, color='purple', linewidth=1, linestyle='--')

plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.title('Loss curve of Poisson Regression')

# Save plot as PDF
plt.savefig(DS1_TRAIN_GRAPH_PATH, format='pdf', bbox_inches='tight')
plt.close()


# read results for dataset 2
with open(DS2_RESULT_PATH, 'r') as file:
        lines = file.readlines()
        # collecting loss
        grad_train_loss = list(map(float, lines[1].split()[1:]))
        grad_val_loss = list(map(float, lines[2].split()[1:]))
        regu_train_loss = list(map(float, lines[4].split()[1:]))
        regu_val_loss = list(map(float, lines[5].split()[1:]))
        adam_train_loss = list(map(float, lines[7].split()[1:]))
        adam_val_loss = list(map(float, lines[8].split()[1:]))
        adamw_train_loss = list(map(float, lines[10].split()[1:]))
        adamw_val_loss = list(map(float, lines[11].split()[1:]))
        adamwd_train_loss = list(map(float, lines[13].split()[1:]))
        adamwd_val_loss = list(map(float, lines[14].split()[1:]))

# Plot train loss graph
# plt.plot(range(len(grad_train_loss)), grad_train_loss, color='blue', linewidth=2, linestyle='-', label="GD")
plt.plot(range(len(regu_train_loss)), regu_train_loss, color='red', linewidth=2, linestyle='-', label="Regularizer")
plt.plot(range(len(adam_train_loss)), adam_train_loss, color='green', linewidth=2, linestyle='-', label="Adam")
# plt.plot(range(len(adamw_train_loss)), adamw_train_loss, color='orange', linewidth=2, linestyle='-', label="AdamW")
plt.plot(range(len(adamwd_train_loss)), adamwd_train_loss, color='purple', linewidth=2, linestyle='-', label="AdamW with Decay")

# Plot validation loss graph
# plt.plot(range(len(grad_val_loss)), grad_val_loss, color='blue', linewidth=1, linestyle='--', label="GD")
plt.plot(range(len(regu_val_loss)), regu_val_loss, color='red', linewidth=1, linestyle='--')
plt.plot(range(len(adam_val_loss)), adam_val_loss, color='green', linewidth=1, linestyle='--')
# plt.plot(range(len(adamw_val_loss)), adamw_val_loss, color='orange', linewidth=1, linestyle='--', label="AdamW")
plt.plot(range(len(adamwd_val_loss)), adamwd_val_loss, color='purple', linewidth=1, linestyle='--')

plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.title('Loss curve of Poisson Regression')

# Save plot as PDF
plt.savefig(DS2_TRAIN_GRAPH_PATH, format='pdf', bbox_inches='tight')