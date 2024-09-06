from typing import Self
from collections.abc import Callable
from random import uniform
from functools import reduce


class Value:
    def __init__(self, data: float, _children: tuple[Self, ...] = ()):
        self.data: float = data
        self.grad: float = 0
        self._backward: Callable[[], None] = lambda: None
        self._pre: tuple[Self, ...] = _children

    def relu(self) -> Self:
        if self.data < 0:
            self.data = 0
        out = Value(self.data, (self,))

        def _backward():
            self.grad += out.grad if self.data > 0 else 0

        out._backward = _backward
        return out

    def backward(self):
        top_order: list[Self] = []
        visited: set[Self] = set()

        def build_toporder(n: Self):
            if n not in visited:
                visited.add(n)
                for i in n._pre:
                    build_toporder(i)
                top_order.append(n)

        build_toporder(self)

        self.grad = 1
        for n in top_order[::-1]:
            n._backward()

    def __repr__(self) -> str:
        return f"Value({self.data})"

    def __add__(self, other: Self) -> Self:
        out: Self = Value(self.data + other.data, _children=(self, other))

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: Self) -> Self:
        out: Self = Value(self.data * other.data, _children=(self, other))

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward
        return out


class Neuron:
    def __init__(self, nin: int):
        self.ws: list[Value] = [Value(uniform(-1, 1)) for _ in range(nin)]
        self.b: Value = Value(0)

    def __call__(self, xs: list[Value]) -> Value:
        return sum((w * x for w, x in zip(self.ws, xs)), self.b)

    def params(self) -> list[Value]:
        return self.ws + [self.b]


class Layer:
    def __init__(self, nin: int, nout: int):
        self.ns: list[Neuron] = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x: list[Value]) -> list[Value]:
        return [n(x) for n in self.ns]

    def params(self) -> list[Value]:
        return reduce(lambda x, y: x + y, [n.params() for n in self.ns], [])


class MLP:
    def __init__(self, nin: int, nouts: list[int]):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x: list[Value]) -> list[Value]:
        for l in self.layers:
            x = l(x)
        return x

    def params(self) -> list[Value]:
        return reduce(lambda x, y: x + y, [l.params() for l in self.layers], [])


if __name__ == "__main__":
    data = [
        [2, 3, -1],
        [3, -1, 0.5],
        [0.5, 1, 1],
        [1, 1, -1],
    ]
    ys = [1, -1, -1, 1]
    data = [[Value(d) for d in subd] for subd in data]

    mlp = MLP(3, [4, 4, 1])

    # learning rate
    lr = 0.005

    for i in range(100):
        yds = [mlp(d)[0] for d in data]
        loss = sum(((Value(-y) + yd) * (Value(-y) + yd) for y, yd in zip(ys, yds)), Value(0))
        # backward
        # reset grad with 0
        for p in mlp.params():
            p.grad = 0
        loss.backward()
        # update
        for p in mlp.params():
            p.data += lr * (-p.grad)
        if (i + 1) % 10 == 0:
            print(f'=====================\niteration {i + 1}', 'loss ', loss.data)
            print([mlp(d)[0].data for d in data])
