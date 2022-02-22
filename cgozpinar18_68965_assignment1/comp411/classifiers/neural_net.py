from __future__ import print_function

from builtins import range
from builtins import object
import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange

class ThreeLayerNet(object):
    """
    A three-layer fully-connected neural network. The net has an input dimension of
    N, hidden layer dimension of H, another hidden layer of dimension H, and performs 
    classification over C classes. We train the network with a softmax loss function 
    and L2 regularization on the weight matrices. The network uses a ReLU nonlinearity 
    after the first and second fully connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-3):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, H)
        b2: Second layer biases; has shape (H,)
        W3: Third layer weights; has shape (H, C)
        b3: Third layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = std * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a three layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        N, D = X.shape

        # Compute the forward pass
        scores = None
        #############################################################################
        # TODO: Perform the forward pass, computing the class scores for the input. #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C).                                                             #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Pass through layer 1
        FC_1_output = np.dot(X, W1) + b1[np.newaxis, :]
        FC_1_activation_output = np.maximum(np.zeros_like(FC_1_output), FC_1_output) # Pass throug ReLU
        
        # Pass through layer 2
        FC_2_output = np.dot(FC_1_activation_output, W2) + b2[np.newaxis, :]
        FC_2_activation_output = np.maximum(np.zeros_like(FC_2_output), FC_2_output)# Pass through ReLU
        
        # Pass through last layer (classification layer)
        FC_3_output = np.dot(FC_2_activation_output, W3) + b3[np.newaxis, :]
        scores = FC_3_output 

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        #############################################################################
        # TODO: Finish the forward pass, and compute the loss. This should include  #
        # both the data loss and L2 regularization for W1 and W2. Store the result  #
        # in the variable loss, which should be a scalar. Use the Softmax           #
        # classifier loss.                                                          #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # handle instability before sotmax activation by subtracting the max score of each corresponding instance before exponentiating the scores
        FC_3_output_stable = FC_3_output - np.max(FC_3_output, axis=1)[:, np.newaxis]
        # pass through softmax activation function
        softmax_output = np.exp(FC_3_output_stable) / np.sum(np.exp(FC_3_output_stable),axis=1)[:, np.newaxis]
        
        # calculate softmax loss (softmax outputs passed into categorical cross-entropy loss)
        cross_entropy_loss = np.sum(-np.log(softmax_output[range(N),y]))
        cross_entropy_loss /= N
        # calculate L2 regularization loss
        l2_loss = reg * np.sum(np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))
        # calculate the total loss
        loss = cross_entropy_loss + l2_loss

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Backward pass: compute gradients
        grads = {}
        #############################################################################
        # TODO: Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # calculate the partial derivative of softmax loss wrt its input
        # convert softmax_scores to be [(softmax_scores[:,j] = softmax_score_of_y_i_th_class - 1) for j = y_i] and
        # [(softmax_scores[:,j] = softmax_score_of_j_th_class) for j != y_i] 
        # dsoftmax_output = (dsoftmax_loss/dFC_3_output)
        softmax_output[range(X.shape[0]), y] = (softmax_output[range(X.shape[0]), y] - 1) # (S - 1) --> Shape(N, C)
        dsoftmax_output = softmax_output * (1 / X.shape[0]) # take the mean by * (1/N)
        
        # dW3 = (dsoftmax_output/dFC_3_output) * (dFC_3_output/dW3) + (dL2_reg/dW3) 
        # NOTE: (dFC_3_output/dW3) = FC_2_activation_output 
        # NOTE: dl2_reg = (dL2_reg/dW3) = lambda * 2 * W3
        dW3 =  np.dot(np.transpose(FC_2_activation_output), dsoftmax_output) # [X * (S - 1)]  --> shape(H, C)
        dl2_reg = 2 * reg * W3
        dW3 += dl2_reg
        # db3 = (dsoftmax_loss/dFC_3_output) * 1
        db3 = np.sum(dsoftmax_output, axis=0) # db3 = (dsoftmax_loss/dFC_3_output) * 1 --> shape(C, )

        # dFC_2_activation_output = (dsoftmax_output/FC_2_activation_output) = (dsoftmax_loss/dFC_3_output) * (dFC_3_output/d)
        # NOTE: (dFC_3_output/dFC_2_activation_output) = W3 
        dFC_2_activation_output = np.dot(dsoftmax_output, W3.T) # (N, C) * (H, C).T = (N, H)
        # dFC_2_output = (dsoftmax_output/dFC_2_output) = (dsoftmax_output/FC_2_activation_output) * (FC_2_activation_output/dFC_2_output) 
        # dRelu = (FC_2_activation_output/dFC_2_output) # NOTE: (dReLU(x)/dx) = {1 if ReLU(x)>0 ; 0 if ReLU(x)=0}
        dReLU = (FC_2_activation_output > 0)
        dFC_2_output = dFC_2_activation_output * dReLU # --> (N, H) .* (N, H) = (N, H) #NOTE: it is not matmul, it is point-wise multiplication

        # dW2 = (dsoftmax_output/dFC_2_output) * (dFC_2_output/dW2)
        # NOTE: (dFC_2_output/dW2) = FC_1_activation_output 
        dW2 = np.dot(FC_1_activation_output.T, dFC_2_output) # (N, H).T * (N, H) = (H, H)
        dl2_reg = 2 * reg * W2
        dW2 += dl2_reg
        # db2 = (dsoftmax_output/dFC_2_output) * (dFC_2_output/db2)
        # NOTE: (dFC_2_output/db2) = 1
        db2 = np.sum(dFC_2_output, axis=0) # (H, )

        # dFC_1_activation_output = (dsoftmax_output/dFC_1_activation_output) = (dsoftmax_output/dFC_2_output) * (dFC_2_output/dFC_1_activation_output)  
        # NOTE: (dFC_2_output/dFC_1_activation_output) = W2
        dFC_1_activation_output = np.dot(dFC_2_output, W2.T) # (N, H) * (H, H) = (N, H)
        # dFC_1_output = (dsoftmax_output/dFC_1_output) = (dsoftmax_output/dFC_1_activation_output) * (dFC_1_activation_output/dFC_1_output)
        # NOTE: dReLU_2 = (dFC_1_activation_output/dFC_1_output) = (dRelu(FC_1_activation)/dFC1_activation) # NOTE: (dReLU(x)/dx) = {1 if ReLU(x)>0 ; 0 if ReLU(x)=0}
        dReLU_2 = (FC_1_activation_output > 0)
        dFC_1_output = dFC_1_activation_output * dReLU_2  # (N, H) .* (N, H) = (N, H)

        # dW1 = (dsoftmax_output/dW1) = (dsoftmax_output/dFC_1_output) * (dFC_1_output/dW1)
        # NOTE: (dFC_1_output/dW1) = X
        dW1 = np.dot(X.T, dFC_1_output) # (D, N) * (N, H) = (D, H)
        dl2_reg = 2 * reg * W1
        dW1 += dl2_reg
        # db1 = (dsoftmax_output/db1) = (dsoftmax_output/dFC_1_output) * (dFC_1_output/db1)
        # NOTE: (dFC_1_output/db1) = 1
        db1 = np.sum(dFC_1_output, axis=0) # (H, )

        # record the calculated parameters in the dict
        grads['W3'] = dW3
        grads['b3'] = db3
        grads['W2'] = dW2
        grads['b2'] = db2
        grads['W1'] = dW1
        grads['b1'] = db1


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                             #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            batch_indices = np.random.choice(range(X.shape[0]), batch_size)
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            self.params['W3'] -= learning_rate * grads['W3']
            self.params['b3'] -= learning_rate * grads['b3']
            self.params['W2'] -= learning_rate * grads['W2']
            self.params['b2'] -= learning_rate * grads['b2']
            self.params['W1'] -= learning_rate * grads['W1']
            self.params['b1'] -= learning_rate * grads['b1']

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
          'loss_history': loss_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this three-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        y_pred = None

        ###########################################################################
        # TODO: Implement this function; it should be VERY simple!                #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        
        y_pred = np.argmax(self.loss(X), axis=1) # predicted class is the index with the highest softmax probability

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred
