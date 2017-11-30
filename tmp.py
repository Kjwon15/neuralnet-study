from nn.main import NeuralNet
from nn.matrix import Matrix

mat = Matrix(init=[[1]])


samples = [
    ([0, 0], [0]),
    ([0, 1], [0]),
    ([1, 0], [0]),
    ([1, 1], [1]),
]

xs, ys = tuple(zip(*samples))

nn = NeuralNet(2, 2, 1)

y_prime = nn.predict([1, 1])
