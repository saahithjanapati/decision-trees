import math
import random
import matplotlib.pyplot as plt
import pygraphviz as pgv

FILENAME = "housevotes84.csv"
global HEADERS
global CURRENT_ID
CURRENT_ID = 0

def reduce_data(data, index, value):
    reduced_data = []
    for elem in data:
        if elem[index] == value: reduced_data.append(elem)
    return reduced_data


def choose_feature(data):
    initial_entropy = calculate_entropy(data)
    information_gain = 0
    reduced_data = []
    best_feature_index = 0
    best_value_set = []
    for index in range(len(data[0])-1):
        value_set = list(set([ elem[index] for elem in data ]))
        new_reduced_data = []
        for value in value_set:
            new_reduced_data.append(reduce_data(data, index, value))
        total_len = 0
        for new_dataset in new_reduced_data: total_len += len(new_dataset)
        expected_entropy = 0
        for new_dataset in new_reduced_data: expected_entropy += len(new_dataset)/total_len * calculate_entropy(new_dataset)
        if initial_entropy - expected_entropy > information_gain:
            information_gain = initial_entropy - expected_entropy
            reduced_data = new_reduced_data
            best_feature_index = index
            best_value_set = value_set
    return best_feature_index, reduced_data, best_value_set


def calculate_entropy(data):
    labels = [entry[-1] for entry in data]
    count_dict = dict.fromkeys(labels, 0)
    for label in labels: count_dict[label] += 1
    entropy = 0
    for key in count_dict.keys():
        prob = count_dict[key]/len(labels)
        entropy += -prob * math.log(prob,2)
    return entropy


class Node:
    def __init__(self, children, values, feature, index, id, decision = None):
        self.children = children
        self.values = values
        self.feature = feature
        self.decision = decision
        self.index = index
        self.id = id

    def print(self):
        print("Children: " + str(self.children))
        print("Values: " + str(self.values))
        print("Feature: " + str(self.feature))
        print("Index: " + str(self.index))
        print("Decision: " + str(self.decision))


def depict_tree(node, G):
    if node.decision is not None:
        G.add_node(node.id, label=node.decision)
    else:
        G.add_node(node.id, label= str(node.feature) + "?")
        for i in range(len(node.children)):
            child = node.children[i]
            depict_tree(child, G)
            G.add_edge(node.id, child.id, label= str(node.values[i]) + "?")


def build_tree(data):
    global CURRENT_ID
    if calculate_entropy(data) == 0:        # generating leaf node
        return_obj = Node([], [], [], -1, CURRENT_ID, data[0][-1])
        CURRENT_ID += 1
        return return_obj
    index, reduced_data, best_value_set =  choose_feature(data)
    # print(index)
    children = [build_tree(reduced_data[i]) for i in range(len(reduced_data))]
    return_obj = Node(children, best_value_set, HEADERS[index], index, CURRENT_ID)
    CURRENT_ID += 1
    return return_obj


def print_tree(node, depth):
    print(" " * depth + "* " + node.feature + "?")

    for i in range(len(node.children)):
        child = node.children[i]
        if child.decision is not None:
            print(" "*(depth+1) + "* " + node.values[i] + " --> " + child.decision)
        else:
            print(" "*(depth+1) + node.values[i]+"?")
            print_tree(node.children[i], depth+2)


def predict(node, vector):
    while node.decision is None:
        value = vector[node.index]
        i = node.values.index(value)
        node = node.children[i]
    return node.decision


def calculate_accuracy(node, test):
    correct = 0
    total = 0
    for vector in test:
        if predict(node, vector[0:-1]) == vector[-1]: correct+=1
        total += 1
    return correct/total

def main():
    global HEADERS
    global CURRENT_ID
    with open(FILENAME) as f:
        input = f.readlines()
    for i in range(len(input)):
        input[i] = input[i].rstrip().split(",")[1:]
    HEADERS = input[0]
    data = input[1:]
    NONMISSING = []
    for input_vector in data:
        if "?" not in input_vector: NONMISSING.append(input_vector)
    test_indices = random.sample(list(range(len(NONMISSING))), 50)
    TEST = [NONMISSING[i] for i in test_indices]
    training_sizes = list(range(5, 183))
    accuracies = []
    train_indices = list(set(list(range(len(NONMISSING)))) - set(test_indices))
    for SIZE in training_sizes:
        TRAIN = []
        new_train_indices = random.sample(train_indices, SIZE)
        for i in range(0, SIZE):
            TRAIN.append(NONMISSING[new_train_indices[i]])
        root_node = build_tree(TRAIN)
        accuracy = calculate_accuracy(root_node, TEST)*100
        accuracies.append(accuracy)


    plt.scatter(training_sizes, accuracies)
    plt.xlabel('Training size')
    plt.ylabel('Accuracy')
    plt.show()

    G = pgv.AGraph(directed=True)
    depict_tree(root_node, G)
    G.layout(prog="dot")
    G.draw("minimal-tree.png")


if __name__ == "__main__": main()
