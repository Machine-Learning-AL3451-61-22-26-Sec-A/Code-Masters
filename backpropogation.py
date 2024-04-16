import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Define the main Streamlit app
def main():
    st.title('Backpropagation Demo')

    # User-defined parameters
    num_layers = st.slider('Number of Layers', min_value=2, max_value=5, value=3)
    layer_sizes = [st.slider(f'Layer {i+1} Size', min_value=1, max_value=10, value=4) for i in range(num_layers)]
    learning_rate = st.slider('Learning Rate', min_value=0.01, max_value=1.0, value=0.1)
    epochs = st.slider('Number of Epochs', min_value=100, max_value=1000, value=500)

    # Define the neural network model using TensorFlow
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(layer_sizes[0],)))
    for size in layer_sizes[1:]:
        model.add(tf.keras.layers.Dense(size, activation='sigmoid'))
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate), loss='mse')

    # Generate random training data
    np.random.seed(42)
    X = np.random.randn(100, layer_sizes[0])
    y = np.random.randint(2, size=(100, layer_sizes[-1]))

    # Train the neural network model
    history = model.fit(X, y, epochs=epochs, verbose=0)

    # Plot the loss curve
    st.subheader('Loss Curve')
    fig, ax = plt.subplots()
    ax.plot(history.history['loss'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    st.pyplot(fig)

    # Display final weights and biases
    st.subheader('Final Weights and Biases')
    for i, layer in enumerate(model.layers[1:]):
        weights, biases = layer.get_weights()
        st.write(f'Layer {i+1} - Weights:')
        st.write(weights)
        st.write(f'Layer {i+1} - Biases:')
        st.write(biases)

if __name__ == "__main__":
    main()
