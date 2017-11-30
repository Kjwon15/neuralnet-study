import math
import random

class NeuralNet():

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.input_nodes = [0] * input_size
        self.hidden_nodes = [0] * hidden_size
        self.output_nodes = [0] * output_size
        self.weight_input = [random.random() for _ in range(input_size * hidden_size)]
        self.weight_output = [random.random() for _ in range(hidden_size * output_size)]

    def feed(self, input_):
        assert len(self.input_nodes) == len(input_)

        for index, val in enumerate(input_):
            self.input_nodes[index] = val

        for index in range(self.hidden_size):
            self.hidden_nodes[index] = self.sigmoid(sum(
                self.input_nodes[i] * self.weight_input[index * self.input_size + i]
                for i in range(self.input_size)
            ))

        for index in range(self.output_size):
            self.output_nodes[index] = self.sigmoid(sum(
                self.hidden_nodes[i] * self.weight_output[index * self.hidden_size + i]
                for i in range(self.hidden_size)
            ))

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))
