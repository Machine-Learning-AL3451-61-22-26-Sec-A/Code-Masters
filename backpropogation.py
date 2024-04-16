# Perform forward pass
def forward_pass(inputs, weights):
    return np.dot(inputs, weights)

# Compute loss
def compute_loss(targets, predictions):
    return np.mean(np.square(targets - predictions))

# Backpropagation
def backpropagation(inputs, targets, weights):
    predictions = forward_pass(inputs, weights)
    loss = compute_loss(targets, predictions)
    
    # Compute gradients of loss with respect to weights
    gradient = -2 * np.dot(inputs.T, (targets - predictions))
    
    return gradient, loss

# Update weights using gradient descent
def update_weights(weights, gradient, learning_rate):
    return weights - learning_rate * gradient

# Training loop
def train(inputs, targets, weights, learning_rate, epochs):
    for epoch in range(epochs):
        gradient, loss = backpropagation(inputs, targets, weights)
        weights = update_weights(weights, gradient, learning_rate)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss}")
    return weights

# Example usage
import numpy as np

# Define training data
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])

# Initialize weights
weights = np.random.rand(2, 1)

# Set learning rate and epochs
learning_rate = 0.1
epochs = 1000

# Train the model
trained_weights = train(inputs, targets, weights, learning_rate, epochs)

# Test the trained model
predictions = forward_pass(inputs, trained_weights)
print("Test Results:")
for i in range(len(inputs)):
    print(f"Input: {inputs[i]}, Output: {predictions[i][0]}")
