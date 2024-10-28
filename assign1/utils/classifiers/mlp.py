"""
Implementation of MLP.
"""

import numpy as np
from utils.layer_utils import DenseLayer, AffineLayer


class MLP:
    """
    MLP (Multilayer Perceptrons) with an arbitrary number of dense hidden layers, and a softmax loss function. 
    For a network with L layers, the architecture will be

    input >> DenseLayer x (L - 1) >> AffineLayer >> softmax_loss >> output

    Here "x (L - 1)" indicate to repeat L - 1 times. 
    """

    def __init__(
        self, input_dim=3072, hidden_dims=[200, 200], num_classes=10, reg=0.0, weight_scale=1e-3
    ):
        """
        Inputs:
        - reg: (float) L2 regularization
        - weight_scale: (float) for layer weight initialization
        """

        self.num_layers = len(hidden_dims) + 1 #3
        dims = [input_dim] + hidden_dims #5800,200,100

        self.layers = [DenseLayer(
            input_dim=dims[i], output_dim=dims[i+1], weight_scale=weight_scale
        ) for i in range(len(dims) - 1)]#2
        self.layers.append(AffineLayer(
            input_dim=dims[-1], output_dim=num_classes, weight_scale=weight_scale
        ))

        #i=0 input = 5800, output = 200 W = np.rand(5800, 200) b = zeros(200) self.params = {'W': W, 'b': b} self.cache = {}  self.gradients = {} self.activation_forward = relu_forward  self.activation_backward = relu_backward

        #i=1 input = 200, output = 100 W = np.rand(200,100) b = zeros(100) self.params = {'W': W, 'b': b} self.cache = {}  self.gradients = {} self.activation_forward = relu_forward  self.activation_backward = relu_backward

        #append input = 100 output = 20 W = np.rand(100,20) b = zeros(20) self.params = {'W': W, 'b': b} self.cache = {}  self.gradients = {} 

        self.reg = reg
        self.velocities = None

    def forward(self, X):
        """
        Feed forward

        Inputs:
        - X: a numpy array of (N, D) containing input data

        Returns a numpy array of (N, K) containing prediction scores
        """

        ############################################################################
        #                            START OF YOUR CODE                            #
        ############################################################################
        ############################################################################
        # TODO: Feedforward                                                        #
        ############################################################################

        for layer in self.layers:
            X = layer.feedforward(X)
            
        #i = 0 out = zeros((X.shape[0], 200) self.cache['A'] = affine output X@W+b; out = self.activation_forward(A);

        #i = 1 out = zeros((X.shape[0], 100) self.cache['A'] = affine output X@W+b; out = self.activation_forward(A);

        #i = 2 out = zeros((X.shape[0], 20) self.cache['A'] = affine output X@W+b;
    
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return X

    def loss(self, scores, labels):
        """
        Calculate the cross-entropy loss and then use backpropogation
        to get gradients wst W,b in each layer.
        Do regularization for better model generalization.

        Inputs:
        - X: input data
        - y: ground truth

        Return loss value(float)
        """

        loss = 0.0

        from ..layer_funcs import softmax_loss

        ############################################################################
        #                            START OF YOUR CODE                            #
        ############################################################################
        ############################################################################
        # TODO: Backpropogation                                                    #
        ############################################################################

        loss, dout = softmax_loss(scores, labels)

        for i in range(len(self.layers) - 1, -1, -1):
            dout = self.layers[i].backward(dout)
            
    
        ############################################################################
        # TODO: Add L2 regularization                                              #
        ############################################################################
        
        squared_weights = 0.0 
        for layer in self.layers:
            squared_weights += np.sum(layer.params['W']**2)
        loss += 0.5 * self.reg * squared_weights
    
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss

    def step(self, learning_rate=1e-2, optim='SGD', momentum=0.0):
        """
        Use SGD to implement a single-step update to each weight and bias.
        Set learning rate to 0.00001.
        """

        # Creates new lists with all parameters and gradients
        # naming rule l{i}_[W, b]: layer i, weights / bias
        params = {
            'l{}_'.format(i + 1) + name: param
            for i, layer in enumerate(self.layers)
            for name, param in layer.params.items()
        }
        grads = {
            'l{}_'.format(i + 1) + name: grad
            for i, layer in enumerate(self.layers)
            for name, grad in layer.gradients.items()
        }
        # Final layout: 
        # params = {
        #     'l1_W': xxx, 'l1_b': xxx, 
        #     'l2_W': xxx, 'l2_b': xxx, 
        #     ..., 
        #     'lN_W': xxx, 'lN_b': xxx, 
        # }
        # grads likewise

        ############################################################################
        # TODO: Use SGD with momentum to update variables in layers                #
        # NOTE: Recall what we did for the TwoLayerNet                             #
        ############################################################################
        ############################################################################
        #                            START OF YOUR CODE                            #
        ############################################################################

        velocities = self.velocities or \
            {name: np.zeros_like(param) for name, param in params.items()}

        reg = self.reg
        grads = {name: grad+ reg * params[name] for name, grad in grads.items()}
        if optim == 'SGD':
            for name, param in params.items():
                params[name] -= learning_rate * grads[name]

        else:
            for name, param in params.items():
                velocities[name] = momentum * velocities[name] + learning_rate * grads[name]
                params[name] -= velocities[name]
        self.update_model((params, reg))
        self.params = params
        self.velocities = velocities
            
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def predict(self, X):
        """
        Return the label prediction of input data

        Inputs:
        - X: (float) a tensor of shape (N, D)

        Returns: 
        - predictions: (int) an array of length N
        """
    
        predictions = np.zeros(X.shape[0])

        ############################################################################
        # TODO:                                                                    #
        # Implement the prediction function.                                       #
        # Think about how the model decides which class to choose.                 #
        ############################################################################
        ############################################################################
        #                             START OF YOUR CODE                           #
        ############################################################################

        scores = self.forward(X)
        predictions = np.argmax(scores, axis = 1)
    
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return predictions

    def update_model(self, model):
        """
        Update layers and reg with new parameters.
        """

        params, reg = model

        for i, layer in enumerate(self.layers):
            layer.update_layer({
                name.split('_')[1]: param for name, param in params.items()
                if name.startswith('l{}'.format(i + 1))
            })

        self.reg = reg

