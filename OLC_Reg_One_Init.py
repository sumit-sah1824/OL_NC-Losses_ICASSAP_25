import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist # type: ignore
from tensorflow.keras.utils import to_categorical  # type: ignore
import matplotlib.pyplot as plt
import pandas as pd

# Set the seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Step 1: Load and preprocess MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype('float32') / 255.0  # Flatten the 28x28 images to 784 vector
y_train = to_categorical(y_train, 10)

# Step 2: Divide dataset into batches of 16
batch_size = 16
num_batches = len(x_train) // batch_size

# Prepare batches
x_batches = [x_train[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]
y_batches = [y_train[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]

# Step 3: Initialize weights W_0 (randomly initialized)
dist = -10 # scaling for the offset
input_dim = x_train.shape[1]
output_dim = y_train.shape[1]
offset = dist * np.random.uniform(0, 1, (input_dim, output_dim))
# print(offset)
W = np.random.randn(input_dim, output_dim) + offset
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

# Projection step to ensure W is within the radius
def project_onto_ball(W, W_init, radius):
    dist = np.linalg.norm(W - W_init)
    if dist > radius:
        return W_init + radius * (W - W_init) / dist
    return W

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

# Step 5: 100 rounds of gradient descent for each batch
W_100_rounds = W.copy()
losses_100_rounds = []
cumulative_losses_100_rounds = []
dist_from_initial = []
for i in range(num_batches):
    for _ in range(300):  # 100 rounds of GD
        W_100_rounds = gradient_descent(x_batches[i], y_batches[i], W_100_rounds)

        Temp = np.linalg.norm(W_100_rounds - W_init)
        dist_from_initial.append(Temp)
    # Calculate loss for the updated weights
    loss_100 = compute_loss(x_batches[i], y_batches[i], W_100_rounds)
    losses_100_rounds.append(loss_100)
    
    # Cumulative loss calculation for 100-round updated weights
    if i == 0:
        cumulative_losses_100_rounds.append(losses_100_rounds[0])
    else:
        cumulative_losses_100_rounds.append(cumulative_losses_100_rounds[-1] + loss_100)

# Calculate the regret
regret = np.array(cumulative_losses) - np.array(cumulative_losses_100_rounds)
radius = 50 #max(dist_from_initial)

# Define different radius values for comparison
radii = [0.1 * radius, 0.2 * radius, 1 * radius]

# Dynamic Regret for each radius
dynamic_regrets = []

for rad in radii:
    W_proj = W_init.copy()
    losses_proj = []
    cumulative_losses_proj = []
    
    for i in range(num_batches):

        # Calculate loss for projected weights
        loss_proj = compute_loss(x_batches[i], y_batches[i], W_proj)
        losses_proj.append(loss_proj)
        
        # Cumulative loss calculation for projected weights
        if i == 0:
            cumulative_losses_proj.append(losses_proj[0])
        else:
            cumulative_losses_proj.append(cumulative_losses_proj[-1] + loss_proj)
        
        # one round of GD
        W_proj = gradient_descent(x_batches[i], y_batches[i], W_proj)
        W_proj = project_onto_ball(W_proj, W_init, rad)  # Project onto the ball of radius `rad`
    
    # Calculate the dynamic regret for this radius
    regret_proj = np.array(cumulative_losses_proj) - np.array(cumulative_losses_100_rounds)
    dynamic_regrets.append(regret_proj)
    # print(type(dynamic_regrets))

# Convert dynamic_regrets (a list of numpy arrays) into a DataFrame
df = pd.DataFrame(dynamic_regrets).T

# Set column names to indicate the corresponding radius
df.columns = [f"Regret_rho_{rad:.2f}" for rad in radii]

# Save to CSV file
df.to_csv("dynamic_regrets.csv", index=False)

# Step 6: Plot the dynamic regret for each radius
plt.figure(figsize=(8, 5), dpi=300)
for idx, rad in enumerate(radii):
    plt.plot(range(1, len(dynamic_regrets[idx]) + 1), dynamic_regrets[idx], label=rf'$\rho$ = {rad:.2f}$(w_1)$')

plt.xlabel(r'Iteration ($T$)')
plt.ylabel(r'Dynamic Regret ($Reg_T$)')
plt.title(r'Dynamic Regret ($Reg_T$) vs Iteration ($T$)')
plt.legend()
plt.grid(True)
plt.show()
