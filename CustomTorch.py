import numpy as np

# Класс тензор для выполнения матричных/векторных операций
class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data)
        self.grad = None
        self.requires_grad = requires_grad

    def __add__(self, other):
        result = Tensor(self.data + other.data)
        return result

    def __matmul__(self, other):
        result = Tensor(self.data @ other.data)
        return result

# Базовый класс модуля
class Module:
    def __init__(self):
        self.training = True

    def train(self, mode=True):
        self.training = mode

    def eval(self):
        self.train(False)

    def forward(self, *args):
        raise NotImplementedError

    def backward(self, *args):
        raise NotImplementedError

    def parameters(self):
        return []

# Реализация линейного слоя из базового класса
class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = Tensor(np.random.randn(in_features, out_features) * 0.01, requires_grad=True)
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)

    def forward(self, x):
        self.input = x
        return Tensor(x.data @ self.weight.data + self.bias.data)

    def backward(self, grad_output):
        self.weight.grad = self.input.data.T @ grad_output.data
        self.bias.grad = np.sum(grad_output.data, axis=0)
        grad_input = grad_output.data @ self.weight.data.T
        return Tensor(grad_input)

    def parameters(self):
        return [self.weight, self.bias]

# Реализация модуля дропаута из базового модуля
class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p  # Dropout probability

    def forward(self, x):
        if self.training:
            # Generate a binary mask: 1 with probability (1-p), 0 with probability p
            mask = np.random.binomial(1, 1 - self.p, size=x.data.shape) / (1 - self.p)
        else:
            # During evaluation, pass input unchanged
            mask = np.ones_like(x.data)
        self.mask = mask
        return Tensor(x.data * self.mask)

    def backward(self, grad_output):
        # Propagate gradients through the same mask
        return Tensor(grad_output.data * self.mask)

# Реализация класса нейронной сети из базового модуля
class NN(Module):
    def __init__(self):
        super().__init__()
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def train(self, mode=True):
        super().train(mode)
        for layer in self.layers:
            layer.train(mode)

    def eval(self):
        self.train(False)

# MSE
class MSELoss:
    def forward(self, y_pred, y_true):
        return Tensor(np.mean((y_pred.data - y_true.data) ** 2))

    def backward(self, y_pred, y_true):
        return Tensor(2 * (y_pred.data - y_true.data) / y_pred.data.size)

# SGD оптимайзер
class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for param in self.parameters:
            if param.grad is not None:
                param.data -= self.lr * param.grad