import streamlit as st
import pandas as pd
import numpy as np

class CandidateElimination:
    def __init__(self, num_features):
        self.num_features = num_features
        self.S = [np.array(['0']*num_features)]  # Initialize specific boundary
        self.G = [np.array(['?']*num_features)]  # Initialize general boundary

    def is_consistent(self, instance, hypothesis):
        for i in range(len(hypothesis)):
            if hypothesis[i] != '?' and hypothesis[i] != instance[i]:
                return False
        return True

    def eliminate(self, instance, label):
        if label == 'Yes':  # Positive instance
            self.G = [g for g in self.G if self.is_consistent(instance, g)]
            for s in list(self.S):
                if not self.is_consistent(instance, s):
                    self.S.remove(s)
                    for i in range(len(s)):
                        if s[i] != instance[i]:
                            new_s = np.copy(s)
                            new_s[i] = '?'
                            self.S.append(new_s)
        else:  # Negative instance
            self.S = [s for s in self.S if self.is_consistent(instance, s)]
            for g in list(self.G):
                if not self.is_consistent(instance, g):
                    self.G.remove(g)
                    for i in range(len(g)):
                        if g[i] != instance[i] and self.S[0][i] == '?':
                            new_g = np.copy(g)
                            new_g[i] = self.S[0][i]
                            self.G.append(new_g)

    def print_hypotheses(self):
        st.write("Specific boundary (S):", self.S)
        st.write("General boundary (G):", self.G)

# Title
st.title("Candidate Elimination Algorithm")

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the data
    data = pd.read_csv(uploaded_file)

    # Display the data
    st.write("Uploaded data:")
    st.write(data)

    # Get number of features
    num_features = len(data.columns) - 1

    # Train the Candidate Elimination algorithm
    ce = CandidateElimination(num_features)
    for index, row in data.iterrows():
        instance = row[:-1].values
        label = row[-1]
        ce.eliminate(instance, label)

    # Print the hypotheses
    st.write("Final hypotheses:")
    ce.print_hypotheses()
