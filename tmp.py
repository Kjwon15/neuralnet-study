from nn.main import NeuralNet
from nn.matrix import Matrix

mat = Matrix(init=[[1]])


samples = [
    ([0, 0], [0]),
    ([0, 1], [1]),
    ([1, 0], [1]),
    ([1, 1], [0]),
]

xs, ys = tuple(zip(*samples))

nn = NeuralNet(2, 12, 1, learning_rate=.1)

for _ in range(10000):
    nn.train(xs, ys)

for x, y in samples:
    print(f'{nn.predict(x)[0]:.7f} : {y[0]:d}')
