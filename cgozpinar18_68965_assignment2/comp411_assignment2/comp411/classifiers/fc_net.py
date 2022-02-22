from builtins import range
from builtins import object
import numpy as np

from comp411.layers import *
from comp411.layer_utils import *


class ThreeLayerNet(object):
    """
    A three-layer fully-connected neural network with Leaky ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of tuple of (H1, H2) yielding the dimension for the
    first and second hidden layer respectively, and perform classification over C classes.

    The architecture should be affine - leakyrelu - affine - leakyrelu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=(64, 32), num_classes=10,
                 weight_scale=1e-3, reg=0.0, alpha=1e-3):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: A tuple giving the size of the first and second hidden layer respectively
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        - alpha: negative slope of Leaky ReLU layers
        """
        self.params = {}
        self.reg = reg
        self.alpha = alpha

        ############################################################################
        # TODO: Initialize the weights and biases of the three-layer net. Weights  #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1', second layer                    #
        # weights and biases using the keys 'W2' and 'b2',                         #
        # and third layer weights and biases using the keys 'W3' and 'b3.          #
        #                                                                          #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        H1 = hidden_dim[0]
        H2 = hidden_dim[1]
        C = num_classes

        self.params['W1'] = np.random.randn(input_dim, H1) * weight_scale
        self.params['b1'] = np.zeros(H1)

        self.params['W2'] = np.random.randn(H1, H2) * weight_scale
        self.params['b2'] = np.zeros(H2)

        self.params['W3'] = np.random.randn(H2, C) * weight_scale
        self.params['b3'] = np.zeros(C)


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer net, computing the  #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # extract params
        W1 = self.params['W1']
        b1 = self.params['b1']

        W2 = self.params['W2']
        b2 = self.params['b2']

        W3 = self.params['W3']
        b3 = self.params['b3']

        alpha = self.alpha


        # flatten the x
        N, D = X.shape[0], np.prod(X.shape[1:])
        X = X.reshape(N, D)

        # pass through first layer(FC + Leaky_ReLU)
        z1, cache_z1 = affine_forward(X, W1, b1)
        z1_leaky_relu, cache_z1_leaky_relu = leaky_relu_forward(z1, {'alpha': alpha})

        # pass through second layer(FC + Leaky_ReLU)
        z2, cache_z2 = affine_forward(z1_leaky_relu, W2, b2)
        z2_leaky_relu, cache_z2_leaky_relu = leaky_relu_forward(z2, {'alpha': alpha})

        # pass through classification layer(FC)
        z3, cache_z3 = affine_forward(z2_leaky_relu, W3, b3)

        scores = z3


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer net. Store the loss#
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # calculate loss
        loss, dz3 = softmax_loss(scores, y)
        # apply regularization
        loss = loss + (0.5 * self.reg * (np.sum(W1*W1) + np.sum(W2*W2) + np.sum(W3*W3)))

        # calculate local gradients through backprop
        # third layer backprop
        dz3 = affine_backward(dz3,cache_z3)
        
        #second layer backprop
        dz2_leaky_relu = leaky_relu_backward(dz3[0], cache_z2_leaky_relu)
        dz2 = affine_backward(dz2_leaky_relu, cache_z2)

        # first layer backprop
        dz1_leaky_relu = leaky_relu_backward(dz2[0], cache_z1_leaky_relu)
        dz1 = affine_backward(dz1_leaky_relu, cache_z1)

        grads['W3'] = dz3[1] + (W3 * self.reg)
        grads['b3'] = dz3[2]

        grads['W2'] = dz2[1] + (W2 * self.reg)
        grads['b2'] = dz2[2]

        grads['W1'] = dz1[1] + (W1 * self.reg)
        grads['b1'] = dz1[2]

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    LeakyReLU nonlinearities, and a softmax loss function. This will also implement
    dropout optionally. For a network with L layers, the architecture will be

    {affine - leakyrelu - [dropout]} x (L - 1) - affine - softmax

    where dropout is optional, and the {...} block is repeated L - 1 times.

    Similar to the ThreeLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, reg=0.0, alpha=1e-2,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - reg: Scalar giving L2 regularization strength.
        - alpha: negative slope of Leaky ReLU layers
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deterministic so we can gradient check the
          model.
        """
        self.use_dropout = dropout != 1
        self.reg = reg
        self.alpha = alpha
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        for i in range(self.num_layers):
          w_name = 'W' + str(i+1)
          b_name = 'b' + str(i+1)
          if i == 0: # first layer
            self.params[w_name] = np.random.randn(input_dim, hidden_dims[i]) * weight_scale
            self.params[b_name] = np.zeros(hidden_dims[i])
          elif i == (self.num_layers - 1): # last layer
            self.params[w_name] = np.random.randn(hidden_dims[i-1], num_classes) * weight_scale
            self.params[b_name] = np.zeros(num_classes)
          else:
            self.params[w_name] = np.random.randn(hidden_dims[i-1], hidden_dims[i]) * weight_scale
            self.params[b_name] = np.zeros(hidden_dims[i])

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as ThreeLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for dropout param since it
        # behaves differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        cache = {}

        z = X

        for i in range(self.num_layers):
          W = self.params['W' + str(i+1)]
          b = self.params['b' + str(i+1)]
          
          if i == (self.num_layers - 1): # last FC layer then pass through softmax too
            z, cache_z = affine_forward(z, W, b)

            # cache for later use in backpropagation
            cache['cache_z' + str(i+1)] = cache_z
              
          else: 
            z, cache_z = affine_forward(z, W, b)
            z, cache_leaky_relu = leaky_relu_forward(z, {'alpha':self.alpha})
            if self.use_dropout:
              z, cache_dropout = dropout_forward(z, self.dropout_param)

            # cache for later use in backpropagation
            cache['cache_z' + str(i+1)] = cache_z
            cache['cache_z' + str(i+1) + 'leaky_relu'] = cache_leaky_relu
            if self.use_dropout:
              cache['cache_dropout' + str(i+1)] = cache_dropout


        scores = z

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss, dz = softmax_loss(scores, y)
        loss += (0.5 * self.reg * np.sum([np.sum(np.square(self.params['W'+ str(i+1)])) for i in range(self.num_layers)])) #add regularization to loss        

        # calculate partial derivatives
        for i in range(self.num_layers, 0, -1):
          W_name = 'W' + str(i)
          b_name = 'b' + str(i)
          
          if i == self.num_layers: # last layer(FC)
            dz, dw, db = affine_backward(dz, cache['cache_z' + str(i)])
            grads[W_name] = dw + (self.reg * self.params[W_name])
            grads[b_name] = db
          else: # (FC - Leaky_ReLU - [Dropout])
            if self.use_dropout:
              dz = dropout_backward(dz, cache['cache_dropout' + str(i)])
            dz = leaky_relu_backward(dz, cache['cache_z' + str(i) + 'leaky_relu'])
            dz, dw, db = affine_backward(dz, cache['cache_z' + str(i)])
            grads[W_name] = dw + (self.reg * self.params[W_name])
            grads[b_name] = db

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
