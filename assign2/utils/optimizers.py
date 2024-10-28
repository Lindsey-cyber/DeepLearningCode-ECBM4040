#!/usr/bin/env/ python
# ECBM E4040 Fall 2024 Assignment 2
# Optimizer implementations

import numpy as np


class Optimizer:
    def __init__(self, model):
        """
        :param model: (class MLP) an MLP model
        """
        self.model = model

    def zero_grad(self):
        """
        Clear all past gradients for the next update
        """

        grads = self.model.grads
        # for each parameter
        for k in grads:
            grads[k] = 0

    def train(
        self, X_train, y_train, X_valid, y_valid,
        num_epoch=10, batch_size=500, learning_rate=1e-3,
        learning_decay=0.95, verbose=False, record_interval=10
    ):
        """
        This function is for MLP model training

        Inputs:
            :param X_train: (float32) input data, a tensor with shape (N, D1, D2, ...)
            :param y_train: (int) label data for classification, a 1D array of length N
            :param X_valid: (float32) input data, a tensor with shape (num_valid, D1, D2, ...)
            :param y_valid: (int) label data for classification, a 1D array of length num_valid
            :param num_epoch: (int) the number of training epochs
            :param batch_size: (int) the size of a single batch for training
            :param learning_rate: (float)
            :param learning_decay: (float) a factor for reducing the learning rate in every epoch
            :param stochastic: (boolean) whether to use stochastic gradient
        """
        model = self.model
        num_train = X_train.shape[0]
        num_batch = num_train // batch_size
        print('number of batches for training: {}'.format(num_batch))

        # recorder
        loss_hist = []
        train_acc_hist = []
        valid_acc_hist = []
        loss = 0.0

        # Loop
        for e in range(num_epoch):
            # Train stage
            model.cnter = 0

            for i in range(num_batch):
                # Batch
                X_batch = X_train[i * batch_size:(i + 1) * batch_size]
                y_batch = y_train[i * batch_size:(i + 1) * batch_size]

                # Clear gradients before each batch
                self.zero_grad()
                self.X_batch = X_batch
                self.y_batch = y_batch

                # Forward
                preds = model.forward(X_batch)
                # Loss
                loss += model.loss(preds, y_batch)

                # Update gradients after each batch
                self.step(learning_rate=learning_rate)

                if (i + 1) % record_interval == 0:
                    loss /= record_interval
                    loss_hist.append(loss)
                    if verbose:
                        print('{}/{} loss: {}'.format(batch_size * (i + 1), num_train, loss))
                    loss = 0.0

            # Validation stage
            train_acc = model.check_accuracy(X_train, y_train)
            val_acc = model.check_accuracy(X_valid, y_valid)
            train_acc_hist.append(train_acc)
            valid_acc_hist.append(val_acc)

            # Shrink learning_rate
            learning_rate *= learning_decay
            print(
                'epoch {}: valid acc = {}, new learning rate = {}, '
                'number of evaluations {}'.format(e + 1, val_acc, learning_rate, model.cnter)
            )

        return loss_hist, train_acc_hist, valid_acc_hist

    def test(self, X_test, y_test, batch_size=10000):
        """
        Inputs:
            :param X_test: (float) a tensor of shape (N, D1, D2, ...)
            :param y_test: (int) an array of length N
            :param batch_size: (int) seperate input data into several batches
        """
        model = self.model
        acc = 0.0
        num_test = X_test.shape[0]

        if num_test <= batch_size:
            acc = model.check_accuracy(X_test, y_test)
            print('accuracy in a small test set: {}'.format(acc))
            return acc

        num_batch = num_test // batch_size
        for i in range(num_batch):
            X_batch = X_test[i * batch_size:(i + 1) * batch_size]
            y_batch = y_test[i * batch_size:(i + 1) * batch_size]
            acc += batch_size * model.check_accuracy(X_batch, y_batch)

        X_batch = X_test[num_batch * batch_size:]
        y_batch = y_test[num_batch * batch_size:]
        if X_batch.shape[0] > 0:
            acc += X_batch.shape[0] * model.check_accuracy(X_batch, y_batch)

        acc /= num_test
        print('test accuracy: {}'.format(acc))
        return acc

    def step(self, learning_rate):
        """
        For the subclasses to implement
        """
        raise NotImplementedError


class SGDOptim(Optimizer):
    def step(self, learning_rate):
        """
        Implement SGD update on network parameters
        
        Inputs:
            :param learning_rate: (float)
        """
        # Get all parameters and their gradients
        params = self.model.params
        grads = self.model.grads

        # Update each parameter
        for k in grads:
            params[k] -= learning_rate * grads[k]


class SGDMomentumOptim(Optimizer):
    def __init__(self, model, momentum=0.5):
        """
        Inputs:
            :param model: a neural netowrk class object
            :param momentum: (float) momentum decay factor
        """
        super().__init__(model)
        self.momentum = momentum
        velocities = dict()
        for k, v in model.params.items():
            velocities[k] = np.zeros_like(v)
        self.velocities = velocities

    def step(self, learning_rate):
        """
        Implement a one-step SGD + Momentum update on network's parameters
        
        Inputs:
            :param learning_rate: (float)
        """
        momentum = self.momentum
        velocities = self.velocities

        # Get all parameters and their gradients
        params = self.model.params
        grads = self.model.grads
        ###################################################
        # TODO: SGD+Momentum, Update params and velocities#
        ###################################################
        
        for k in grads:
            velocities[k] = momentum * velocities[k] + grads[k]
            params[k] -= learning_rate * velocities[k]
        
        
        ###################################################
        #               END OF YOUR CODE                  #
        ###################################################


class SGDNestMomentumOptim(SGDMomentumOptim):
    
    def step(self, learning_rate):
        """
        Implement a one-step SGD + Nesterov Momentum update on network's parameters
        
        Inputs:
            :param learning_rate: (float)
        """
        momentum = self.momentum
        velocities = self.velocities

        # Get all parameters and their gradients
        params = self.model.params
        grads = self.model.grads
        ###################################################
        # TODO: SGD+Momentum, Update params and velocities#
        ###################################################

        for k in grads:
            velocities[k] = momentum * velocities[k] + grads[k]
            velocities[k] = momentum * velocities[k] + grads[k]
            params[k] -= learning_rate * velocities[k]
        
        ###################################################
        #               END OF YOUR CODE                  #
        ###################################################

        
        
class AdamOptim(Optimizer):
    def __init__(self, model, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        Inputs:
            :param model: a neural network class object
            :param beta1: (float) should be close to 1
            :param beta2: (float) similar to beta1
            :param eps: (float) in different case, the good value for eps will be different
        """
        super().__init__(model)

        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        # Initialize m0 and v0
        self.mean, self.var = {}, {}
        for k, v in model.params.items():
            self.mean[k] = np.zeros_like(v)
            self.var[k] = np.zeros_like(v)

        # Time step
        self.t = 0

    def step(self, learning_rate):
        """
        Implement a one-step Adam update on network's parameters

        Inputs:
            :param learning_rate: (float)
        """
        # Update time step
        self.t += 1

        # Get params
        model = self.model
        beta1 = self.beta1
        beta2 = self.beta2
        eps = self.eps
        t = self.t

        # Stored moments
        mean = self.mean
        var = self.var

        # Get all parameters and their gradients
        params = model.params
        grads = model.grads
        ###################################################
        # TODO: Adam, Update mean and variance and params #
        ###################################################

        for k in grads:
            mean[k] = beta1 * mean[k] + (1 - beta1) * grads[k]
            var[k] = beta2 * var[k] + (1 - beta2) * grads[k] * grads[k]
            mean_hat = mean[k] / (1 - np.power(beta1, t))
            var_hat = var[k] / (1 - np.power(beta2, t))
            params[k] -= learning_rate * np.divide(mean_hat, np.sqrt(var_hat + eps))
        
        ###################################################
        #               END OF YOUR CODE                  #
        ###################################################

        

class AdamWOptim(Optimizer):
    def __init__(self, model, beta1=0.9, beta2=0.999, eps=1e-8, l2_coeff = 1):
        """
        Inputs:
        :param model: a neural network class object
        :param beta1: (float) should be close to 1
        :param beta2: (float) similar to beta1
        :param eps: (float) in different case, the good value for eps will be different
        """
        self.model = model
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.l2_coeff = l2_coeff

        momentums = dict()
        velocities = dict()
        for k, v in model.params.items():
            momentums[k] = np.zeros_like(v)
            velocities[k] = np.zeros_like(v)
        self.momentums = momentums
        self.velocities = velocities
        self.t = 0

    def step(self, learning_rate):
        """
        Implement a one-step Adam update on network's parameters
        
        Inputs:
        :param model: a neural network class object
        :param learning_rate: (float)
        """
        beta1 = self.beta1
        beta2 = self.beta2
        eps = self.eps
        l2_coeff = self.l2_coeff
        model = self.model

        momentums = self.momentums
        velocities = self.velocities
        t = self.t
        # create two new dictionaries containing all parameters and their gradients
        params, grads = model.params, model.grads
        ###################################################
        # TODO: Adam W, Update t, momentums, velocities   #
        # and params                                      #
        ###################################################
        
        raise NotImplementedError

        ###################################################

        
class RMSpropOptim(Optimizer):
    def __init__(self, model, gamma=0.9, eps=1e-12):
        """
        Inputs:
        :param model: a neural network class object
        :param gamma: (float) suggest to be 0.9
        :param eps: (float) a small number
        """
        self.gamma = gamma
        self.eps = eps
        cache = dict()
        for k, v in model.params.items():
            cache[k] = np.zeros_like(v)
        self.cache = cache
        self.model = model

    def step(self, learning_rate):
        """
        Implement a one-step RMSprop update on network's parameters
        And a good default learning rate can be 0.001.
        
        Inputs:
        :param model: a neural network class object
        :param learning_rate: (float)
        """
        gamma = self.gamma
        eps = self.eps
        cache = self.cache        
        model = self.model
        # create two new dictionaries containing all parameters and their gradients
        params, grads = model.params, model.grads
        ###################################################
        # TODO: RMSprop, Update params and                #
        # cache (moving average of squared gradients)     #
        ###################################################
        raise NotImplementedError
        
        ###################################################
        #               END OF YOUR CODE                  #
        ###################################################