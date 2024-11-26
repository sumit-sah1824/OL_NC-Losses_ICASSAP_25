import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist  # type: ignore
from tensorflow.keras.utils import to_categorical  # type: ignore
import matplotlib.pyplot as plt

# Set the seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Step 1: Load and preprocess MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Select only 10,000 samples from the training set
x_train = x_train[:60000].reshape(-1, 28*28).astype('float32') / 255.0  # Flatten the 28x28 images to 784 vector
y_train = to_categorical(y_train[:60000], 10)

# Step 2: Prepare batches with 60% repeated data and 40% unique data
batch_size = 32
num_batches = len(x_train) // batch_size
repeat_ratio = 0.70

# Step 2.1: Select 60% of the dataset to repeat in every batch
num_repeated = int(batch_size * repeat_ratio)
x_repeated = x_train[:num_repeated]
y_repeated = y_train[:num_repeated]

# Step 2.2: Prepare batches with 60% repeated data and 40% unique data
x_batches = []
y_batches = []

for i in range(num_batches):
    start_idx = num_repeated + i * (batch_size - num_repeated)
    x_unique = x_train[start_idx:start_idx + (batch_size - num_repeated)]
    y_unique = y_train[start_idx:start_idx + (batch_size - num_repeated)]
    
    x_batch = np.vstack([x_repeated, x_unique])
    y_batch = np.vstack([y_repeated, y_unique])
    
    x_batches.append(x_batch)
    y_batches.append(y_batch)

# Step 3: Initialize weights W_0 (randomly initialized)
input_dim = x_train.shape[1]
output_dim = y_train.shape[1]
W = np.random.randn(input_dim, output_dim)
W_init = W.copy()

# Store losses and cumulative losses
losses = []
cumulative_losses = []

# Define a simple loss function (cross-entropy)
def compute_loss(X, y, W):
    logits = np.dot(X, W)
    probs = tf.nn.softmax(logits).numpy()
    return -np.mean(np.sum(y * np.log(probs), axis=1))

# Define a gradient descent update
def gradient_descent(X, y, W, lr=0.01):
    logits = np.dot(X, W)
    probs = tf.nn.softmax(logits).numpy()
    grad = np.dot(X.T, (probs - y)) / len(X)
    return W - lr * grad

# Step 4: Iterate through each batch
for i in range(num_batches):
    # Step 4.1: Calculate the loss for the current batch
    loss = compute_loss(x_batches[i], y_batches[i], W)
    losses.append(loss)

    # Step 4.2: Cumulative loss calculation
    if i == 0:
        cumulative_losses.append(losses[0])
    else:
        cumulative_losses.append(cumulative_losses[-1] + loss)

    # Step 4.3: Update weights using one round of gradient descent
    W = gradient_descent(x_batches[i], y_batches[i], W)

# Step 5: Calculate variation and cumulative variation
variation = []
for i in range(len(cumulative_losses) - 1):
    var = cumulative_losses[i + 1] - cumulative_losses[i]
    variation.append(var)

# Sum_Variation Calculation
cumulative_variation = []
current_sum = 0
for i in range(len(variation)):
    current_sum += variation[i]
    cumulative_variation.append(current_sum)

# Create a figure with 300 DPI
plt.figure(figsize=(8, 5), dpi=300)

# Plot the cumulative variation
plt.plot(cumulative_variation)

# Add labels and title
plt.xlabel(r'Iteration $(T)$')
plt.ylabel(r'Sum Variation $(\sum_{i=1}^TV_t)$')
# plt.title(r'Sum Variation $(\sum_{i=1}^TV_t)$ vs Iteration ($T$)')

# Show grid
plt.grid(True)

# Show the plot
plt.show()

# plt.figure(figsize=(8, 5), dpi=300)

# # Plot the variation
# plt.plot(variation)

# # Add labels and title
# plt.xlabel(r'Iteration ($T$)')
# plt.ylabel(r'Variation ($V_T$)')
# plt.title(r'Variation ($V_T$) vs Iteration ($T$)')

# # Show grid
# plt.grid(True)

# # Show the plot
# plt.show()
