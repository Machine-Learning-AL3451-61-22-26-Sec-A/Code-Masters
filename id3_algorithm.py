import math
import csv
import streamlit as st

class Node:
    def __init__(self, attribute):
        self.attribute = attribute
        self.children = []
        self.answer = ""

def load_csv(filename):
    lines = csv.reader(open(filename, "r"))
    dataset = list(lines)
    headers = dataset.pop(0)
    return dataset, headers

def subtables(data, col, delete): 
    dic = {}
    coldata = [row[col] for row in data]
    attr = list(set(coldata))
    for k in attr:
        dic[k] = []

    for y in range(len(data)):
        key = data[y][col]
        if delete:
            del data[y][col]
        dic[key].append(data[y])
    return attr, dic

def entropy(S):
    attr = list(set(S))
    if len(attr) == 1:
        return 0

    counts = [0, 0]
    for i in range(2):
        counts[i] = sum([1 for x in S if attr[i] == x]) / (len(S) * 1.0)

    sums = 0
    for cnt in counts:
        sums += -1 * cnt * math.log(cnt, 2)
    return sums

def compute_gain(data, col):
    attValues, dic = subtables(data, col, delete=False)
    total_entropy = entropy([row[-1] for row in data])
    for x in range(len(attValues)):
        ratio = len(dic[attValues[x]]) / (len(data) * 1.0)
        entro = entropy([row[-1] for row in dic[attValues[x]]]) 
        total_entropy -= ratio * entro

    return total_entropy
 
def build_tree(data, features):
    lastcol = [row[-1] for row in data]
    if len(set(lastcol)) == 1:
        node = Node("")
        node.answer = lastcol[0]
        return node

    n = len(data[0]) - 1
    gains = [compute_gain(data, col) for col in range(n)]

    split = gains.index(max(gains))
    node = Node(features[split])
    fea = features[:split] + features[split + 1:]

    attr, dic = subtables(data, split, delete=True)
    for x in range(len(attr)):
        child = build_tree(dic[attr[x]], fea) 
        node.children.append((attr[x], child))

    return node

def print_tree(node, level, tree_str):
    if node.answer != "":
        tree_str += "       " * level + node.answer + "\n"
        return tree_str

    tree_str += "       " * level + node.attribute + "\n"
    for value, n in node.children:
        tree_str += "       " * (level + 1) + value + "\n"
        tree_str = print_tree(n, level + 2, tree_str)

    return tree_str

def classify(node, x_test, features):
    if node.answer != "":
        return node.answer

    pos = features.index(node.attribute)
    for value, n in node.children:
        if x_test[pos] == value:
            return classify(n, x_test, features)

# Streamlit web app
def main():
    st.title('Decision Tree Classifier')

    dataset, features = load_csv("PlayTennis.csv")
    node = build_tree(dataset, features)

    st.subheader('Decision Tree:')
    tree_str = print_tree(node, 0, "")
    st.text(tree_str)

    testdata, features = load_csv("PlayTennisTestData.csv")
    st.subheader('Predictions:')
    for xtest in testdata:
        st.text(f"The test instance: {xtest}")
        prediction = classify(node, xtest, features)
        st.text(f"The predicted label: {prediction}")

if __name__ == "__main__":
    main()
