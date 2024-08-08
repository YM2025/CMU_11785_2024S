# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import pdb

class Dropout(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x, train=True):

        if train:
            # Generate mask and apply to x
            self.mask = np.random.binomial(1, 1-self.p, x.shape[0]*x.shape[1]).reshape(x.shape)
            return x * self.mask * 1/(1-self.p)
            
        else:
            # Return x as is
            return x
		
    def backward(self, delta):
        # Multiply mask with delta and return
        return delta * self.mask