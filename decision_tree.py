import math
FILENAME = "mushroom.csv"
global HEADERS

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
    def __init__(self, children, values, feature, decision = None):
        self.children = children
        self.values = values
        self.feature = feature
        self.decision = decision

    def print(self):
        print("Children: " + str(self.children))
        print("Values: " + str(self.values))
        print("Feature: " + self.feature)
        print("Decision: " + str(self.decision))


def build_tree(data):
    if calculate_entropy(data) == 0:        # generating leaf node
        return Node([], [], [], data[0][-1])
    index, reduced_data, best_value_set =  choose_feature(data)
    children = [build_tree(reduced_data[i]) for i in range(len(reduced_data))]
    return Node(children, best_value_set, HEADERS[index])


def print_tree(node, depth):
    print(" " * depth + "* " + node.feature + "?")

    for i in range(len(node.children)):
        child = node.children[i]
        if child.decision is not None:
            print(" "*(depth+1) + "* " + node.values[i] + " --> " + child.decision)
        else:
            print(" "*(depth+1) + node.values[i]+"?")
            print_tree(node.children[i], depth+2)


def main():
    global HEADERS
    with open(FILENAME) as f:
        input = f.readlines()
    for i in range(len(input)):
        input[i] = input[i].rstrip().split(",")
    HEADERS = input[0]
    data = input[1:]
    root_node = build_tree(data)
    print_tree(root_node, 0)


if __name__ == "__main__": main()
