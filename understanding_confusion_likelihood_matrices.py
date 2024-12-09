import numpy as np
import torch
from matplotlib import pyplot as plt
import seaborn as sns

outputs = torch.Tensor(np.random.rand(100, 3, 1))

def create_normalized_gaussian_array(length, center_index, std_dev=1):
    x = np.arange(length)
    gaussian_array = np.exp(-(x - center_index) ** 2 / (2 * std_dev ** 2))
    gaussian_array /= gaussian_array.sum()  # Normalize the array to sum to 1
    return gaussian_array.reshape(length, 1)

def create_labels():
    labels = np.zeros((100, 1, 3))
    for i in range(labels.shape[0]):
        labels[i][0][np.random.randint(0, labels.shape[2])] = 1
    return labels


def plot_confusion_matrix(matrix, title, cmap='Blues'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, cmap=cmap, cbar=True, fmt=".2f")
    plt.title(title)
    plt.xlabel('True Class')
    plt.ylabel('Predicted Class')
    plt.show()

n = 10
elements = 1000

labels = np.zeros((elements, 1, n))
simulated_classifier_outputs = np.zeros((elements, n, 1))

for i in range(elements):
    rand_index = np.random.randint(0, n)

    simulated_classifier_outputs[i] = (create_normalized_gaussian_array(n, rand_index))

    label_array = np.zeros((1,n))
    label_array[0][rand_index] = 1
    labels[i] = label_array

confusion_likelihood_matrices = torch.bmm(torch.Tensor(simulated_classifier_outputs), torch.Tensor(labels))
#print(confusion_likelihood_matrices)
confusion_likelihood_matrix = torch.sum(confusion_likelihood_matrices, 0)

confusion_likelihood_matrix_column_sums = np.sum(confusion_likelihood_matrix.numpy(), axis=0, keepdims=True)
print(f"Column sums of confusion likelihood matrix: {confusion_likelihood_matrix_column_sums}")

normalized_confusion_likelihood_matrix = confusion_likelihood_matrix / confusion_likelihood_matrix_column_sums
normalized_confusion_likelihood_matrix_column_sums = np.sum(normalized_confusion_likelihood_matrix.numpy(), axis=0, keepdims=True)
print(f"Column sums of normalized confusion likelihood matrix: {normalized_confusion_likelihood_matrix_column_sums}")

print(confusion_likelihood_matrix)
print(normalized_confusion_likelihood_matrix)

plot_confusion_matrix(confusion_likelihood_matrix.numpy(), "Confusion Likelihood Matrix")
plot_confusion_matrix(normalized_confusion_likelihood_matrix.numpy(), "Normalized Confusion Likelihood Matrix")
