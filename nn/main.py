import logging.config
import math
import os
import random

from .matrix import Matrix

logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'brief': {
            'format': '%(asctime)s:%(name)s:%(levelname)s:%(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S %z',
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'brief',
            'level': 'DEBUG',
        }
    },
    'loggers': {
        'NeuralNet': {
            'handlers': ['console'],
            'level': os.environ.get('LOGLEVEL', 'WARNING').upper(),
            'propagate': False,
        },
    },
    'root': {
        'level': 'WARNING',
        'handlers': ['console'],
    },
})


class NeuralNet():

    logger = logging.getLogger('NeuralNet')

    def __init__(self, input_size: int, hidden_size: int, output_size: int, learning_rate=0.2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.input_nodes = Matrix(1, input_size)
        self.w1 = Matrix(init=[[random.random()] * hidden_size for _ in range(input_size)])
        self.hidden_nodes = Matrix(1, hidden_size)
        self.w2 = Matrix(init=[[random.random()] * output_size for _ in range(hidden_size)])
        self.output_nodes = Matrix(1, output_size)
        self.output = None

    def predict(self, x):
        self.feed(x)
        return self.sigmoid(self.output_nodes).array[0]

    def feed(self, input_):
        assert len(input_) == self.input_nodes.xsize

        for index, val in enumerate(input_):
            self.input_nodes[0, index] = val

        self.hidden_nodes = self.input_nodes * self.w1
        self.output_nodes = self.sigmoid(self.hidden_nodes) * self.w2
        self.output = self.sigmoid(self.output_nodes)

    def train(self, xs, ys):
        xs = Matrix(init=xs)
        ys = Matrix(init=ys)
        hidden_nodes = xs * self.w1  # multiple
        hidden_output = self.sigmoid(hidden_nodes)
        output_nodes = hidden_output * self.w2
        result = self.sigmoid(output_nodes)

        output_error = ys - self.sigmoid(output_nodes)
        self.logger.debug(output_error.array[0])
        output_delta = output_error @ self.dsigmoid(result)

        hidden_error = output_delta * self.w2.transpose
        hidden_delta = hidden_error @ self.dsigmoid(hidden_output)

        self.w1 += xs.transpose * hidden_delta @ self.learning_rate
        self.w2 += hidden_output.transpose * output_delta @ self.learning_rate

    @staticmethod
    def sigmoid(mat: Matrix):
        newmat = Matrix(init=[
            [1 / (1 + math.exp(-mat[y, x])) for x in range(mat.xsize)]
            for y in range(mat.ysize)
        ])
        return newmat

    @staticmethod
    def dsigmoid(mat: Matrix):
        return Matrix(init=[
            [mat[y, x] * (1 - mat[y, x]) for x in range(mat.xsize)]
            for y in range(mat.ysize)
        ])
