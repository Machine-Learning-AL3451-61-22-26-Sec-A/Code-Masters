import streamlit as st
import numpy as np

# Backpropagation algorithm
def backpropagation(inputs, targets, weights, learning_rate, epochs):
    for epoch in range(epochs):
        predictions = forward_pass(inputs, weights)
        loss = compute_loss(targets, predictions)
        gradient = -2 * np.dot(inputs.T, (targets - predictions))
        weights = update_weights(weights, gradient, learning_rate)
    return weights

def forward_pass(inputs, weights):
    return np.dot(inputs, weights)

def compute_loss(targets, predictions):
    return np.mean(np.square(targets - predictions))

def update_weights(weights, gradient, learning_rate):
    return weights - learning_rate * gradient

# Streamlit app
st.title('Neural Network Backpropagation')
st.sidebar.header('Training Parameters')

# Input parameters
learning_rate = st.sidebar.slider('Learning Rate', min_value=0.01, max_value=1.0, value=0.1, step=0.01)
epochs = st.sidebar.slider('Epochs', min_value=100, max_value=10000, value=1000, step=100)

# Training data
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])

# Train the model
weights = backpropagation(inputs, targets, np.random.rand(2, 1), learning_rate, epochs)

# Test the trained model
st.subheader('Test Results:')
for i in range(len(inputs)):
    output = forward_pass(inputs[i], weights)
    st.write(f"Input: {inputs[i]}, Output: {output[0]}")
