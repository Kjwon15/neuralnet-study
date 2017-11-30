from main import NeuralNet


samples = [
    ([0, 0], [0]),
    ([0, 1], [0]),
    ([1, 0], [0]),
    ([1, 1], [1]),
]

nn = NeuralNet(2, 2, 1)

y_prime = nn.predict([1, 1])
