import numpy as np

from mytorch.nn.linear import Linear
from mytorch.nn.activation import ReLU


class MLP0:

    def __init__(self, debug=False):
        self.layers = [Linear(2, 3), ReLU()]
        self.debug = debug

    def forward(self, A0):
        
        Z0 = self.layers[0].forward(A0)  # TODO
        A1 = self.layers[1].forward(Z0)  # TODO

        if self.debug:

            self.Z0 = Z0
            self.A1 = A1

        return A1

    def backward(self, dLdA1):

        dLdZ0 = self.layers[1].backward(dLdA1)  # TODO
        dLdA0 = self.layers[0].backward(dLdZ0)  # TODO

        if self.debug:

            self.dLdZ0 = dLdZ0
            self.dLdA0 = dLdA0

        return dLdA0


class MLP1:

    def __init__(self, debug=False):
        
        self.l0 = Linear(2, 3)
        self.f0 = ReLU()
        self.l1 = Linear(3, 2)
        self.f1 = ReLU()
        
        self.layers = [self.l0, self.f0, self.l1, self.f1]

        self.debug = debug

    def forward(self, A0):
        
        Z0 = self.l0.forward(A0)  # TODO
        A1 = self.f0.forward(Z0)  # TODO

        Z1 = self.l1.forward(A1)# TODO
        A2 = self.f1.forward(Z1)  # TODO

        if self.debug:
            self.Z0 = Z0
            self.A1 = A1
            self.Z1 = Z1
            self.A2 = A2

        return A2

    def backward(self, dLdA2):
        
        dLdZ1 = self.f1.backward(dLdA2)  # TODO
        dLdA1 = self.l1.backward(dLdZ1)  # TODO

        dLdZ0 = self.f0.backward(dLdA1)  # TODO
        dLdA0 = self.l0.backward(dLdZ0)  # TODO

        if self.debug:

            self.dLdZ1 = dLdZ1
            self.dLdA1 = dLdA1

            self.dLdZ0 = dLdZ0
            self.dLdA0 = dLdA0

        return dLdZ0


class MLP4:
    def __init__(self, debug=False):

        # List of Hidden and activation Layers in the correct order
        self.layers = [
            Linear(2, 4),
            ReLU(),
            Linear(4, 8),
            ReLU(),
            Linear(8, 8),
            ReLU(),
            Linear(8, 4),
            ReLU(),
            Linear(4, 2), 
            ReLU()   
        ]  
        
        self.L = len(self.layers)

        self.debug = debug

    def forward(self, A):

        if self.debug:
            self.A = [A]

        for i in range(self.L):

            A = self.layers[i].forward(A)  

            if self.debug:

                self.A.append(A)

        return A

    def backward(self, dLdA):

        if self.debug:

            self.dLdA = [dLdA]

        L = len(self.layers)

        for i in reversed(range(L)):

            dLdA = self.layers[i].backward(dLdA)

            if self.debug:

                self.dLdA = [dLdA] + self.dLdA

        return dLdA
