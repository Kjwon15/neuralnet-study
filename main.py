import math
import random

from matrix import Matrix


class NeuralNet():

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.input_nodes = Matrix(1, input_size)
        self.w1 = Matrix(input_size, hidden_size, [[random.random()] * hidden_size for _ in range(input_size)])
        self.hidden_nodes = Matrix(1, hidden_size)
        self.w2 = Matrix(hidden_size, output_size, [[random.random()] * output_size for _ in range(hidden_size)])
        self.output_nodes = Matrix(1, output_size)

    def predict(self, x):
        self.feed(x)
        return self.output_nodes.array[0]

    def feed(self, input_):
        assert self.input_nodes.xsize == len(input_)

        for index, val in enumerate(input_):
            self.input_nodes[0, index] = val

        self.hidden_nodes = self.input_nodes * self.w1
        self.output_nodes = self.hidden_nodes * self.w2

    @staticmethod
    def sigmoid(x: float):
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def dsigmoid(y: float):
        return y * (1 - y)
