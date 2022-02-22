from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg, regtype='L2'):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength
    - regtype: Regularization type: L1 or L2

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization! Implement both L1 and L2 regularization based on the      #
    # parameter regtype.                                                        #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    if regtype == 'L2':
      # Calculate y_pred = X * W for each x_i
      for i, x_i in enumerate(X): # loop through every xi in X
        y_i = y[i] # correct class label for x_i
        scores_i = np.dot(x_i, W) # (N, C) shape
        # handle numerical instability by shifting the values by the highest score
        scores_i -= np.max(scores_i)
        # converts scores_i to softmax output probabilities
        softmax_scores_i = np.exp(scores_i) / np.sum(np.exp(scores_i))
        # calculate softmax loss
        softmax_loss_i = - np.log(np.exp(scores_i[y_i]) / np.sum(np.exp(scores_i)))
        loss += softmax_loss_i

        # calculate derivatives (dW)
        for j in range(W.shape[1]):# for each class score update dW's j'th column
          if j == y_i: # (softmax_loss - 1) * X
            dW[:, y_i] += (softmax_scores_i[y_i] - 1) * X[i].T
          else: # softmax_scores * X
            dW[:, j] += (softmax_scores_i[j]) * X[i].T
        
        

        
      # take the mean of total loss
      loss /= X.shape[0] # loss = 1/N sum(Li)
      # calculate the l2 regularization loss (i.e lambda * R(W))
      l2_loss = reg * np.sum(np.square(W))
      loss += l2_loss

      # scale dW by 1/N
      dW /= X.shape[0]
      # calculate the gradient of L2 regularization (lambda * 2 * W)
      dW_l2 = reg * 2 * W
      # Sum all of the gradients(dL2/dW + dsoftmax/dW)
      dW += dW_l2

    else:# L1 regularization
       # Calculate y_pred = X * W for each x_i
      for i, x_i in enumerate(X): # loop through every xi in X
        y_i = y[i] # correct class label for x_i
        scores_i = np.dot(x_i, W) # (N, C) shape
        # handle numerical instability by shifting the values by the highest score
        scores_i -= np.max(scores_i)
        # converts scores_i to softmax output probabilities
        softmax_scores_i = np.exp(scores_i) / np.sum(np.exp(scores_i))
        # calculate softmax loss
        softmax_loss_i = - np.log(np.exp(scores_i[y_i]) / np.sum(np.exp(scores_i)))
        loss += softmax_loss_i

        # calculate derivatives (dW)
        for j in range(W.shape[1]):# for each class score update dW's j'th column
          if j == y_i: # (softmax_score_of_true_class_label - 1) * X
            dW[:, y_i] += (softmax_scores_i[y_i] - 1) * X[i].T
          else: # (softmax_scores_of_j_th_class_label) * X
            dW[:, j] += (softmax_scores_i[j]) * X[i].T
        

        
      # take the mean of total loss
      loss /= X.shape[0] # loss = 1/N sum(Li)
      # calculate the l2 regularization loss (i.e lambda * R(W))
      l1_loss = reg * np.sum(np.abs(W))
      loss += l1_loss

      # scale dW by 1/N
      dW /= X.shape[0]
      # calculate the gradient of L2 regularization (sign of W)
      dW_l1 = reg * (np.abs(W) / W)
      # Sum all of the gradients(dL2/dW + dsoftmax/dW)
      dW += dW_l1

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg, regtype='L2'):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization! Implement both L1 and L2 regularization based on the      #
    # parameter regtype.                                                        #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    if regtype == 'L2':
      # Calculate y_pred = X * W using broadcasting
      scores = np.dot(X, W) # (N, C) shape
      # handle numerical instability by shifting the values of each row of scores by the highest score of the corresponding row(i.e each scores vector)
      scores -= np.max(scores, axis=1)[:, np.newaxis] # add new dimension for broadcasting 
      # convert scores values to softmax values(i.e probability distribution)
      scores = np.exp(scores) # numerator of softmax equation
      scores_denominator = np.sum(scores, axis=1)[:, np.newaxis] # denominator of softmax equation (shape (N,1))
      softmax_scores = scores / scores_denominator # scores becomes the softmax outputs (broadcasting is used here)
      
      # calculate the softmax loss(i.e. multi-class cross entropy loss on the outputs of softmax activation function)
      softmax_loss = np.sum(- np.log(softmax_scores[range(X.shape[0]), y]))
      softmax_loss /= X.shape[0] #average the individual sample x_i losses
      # calculate L2 regularization loss
      l2_loss = reg * np.sum(np.square(W), axis=None)
      # update the total loss
      loss = softmax_loss + l2_loss
      
      # calculate the partial derivative of softmax loss wrt W
      # convert softmax_scores to be [(softmax_scores[:,j] = softmax_score_of_y_i_th_class - 1) for j = y_i] and
      # [(softmax_scores[:,j] = softmax_score_of_j_th_class) for j != y_i] 
      softmax_scores[range(X.shape[0]), y] = softmax_scores[range(X.shape[0]), y] - 1 
      dW = (1 / X.shape[0]) * np.dot(np.transpose(X), softmax_scores) # [X * (S - 1)] / N
      # calculate the partial derivative of L2 regularization wrt W (lambda * 2 * W)
      dW_l2 = reg * 2 * W
      # Sum all of the gradients(dL2/dW + dsoftmax/dW)
      dW += dW_l2

    else:# L1 regularization
      # Calculate y_pred = X * W using broadcasting
      scores = np.dot(X, W) # (N, C) shape
      # handle numerical instability by shifting the values of each row of scores by the highest score of the corresponding row(i.e each scores vector)
      scores -= np.max(scores, axis=1)[:, np.newaxis] # add new dimension for broadcasting 
      # convert scores values to softmax values(i.e probability distribution)
      scores = np.exp(scores) # numerator of softmax equation
      scores_denominator = np.sum(scores, axis=1)[:, np.newaxis] # denominator of softmax equation (shape (N,1))
      softmax_scores = scores / scores_denominator # scores becomes the softmax outputs (broadcasting is used here)
      
      # calculate the softmax loss(i.e. multi-class cross entropy loss on the outputs of softmax activation function)
      softmax_loss = np.sum(- np.log(softmax_scores[range(X.shape[0]), y]))
      softmax_loss /= X.shape[0] #average the individual sample x_i losses
      # calculate L2 regularization loss
      l1_loss = reg * np.sum(np.abs(W), axis=None)
      # update the total loss
      loss = softmax_loss + l1_loss
      
      # calculate the partial derivative of softmax loss wrt W
      # convert softmax_scores to be [(softmax_scores[:,j] = softmax_score_of_y_i_th_class - 1) for j = y_i] and
      # [(softmax_scores[:,j] = softmax_score_of_j_th_class) for j != y_i] 
      softmax_scores[range(X.shape[0]), y] = softmax_scores[range(X.shape[0]), y] - 1 # (S - 1)
      dW = (1 / X.shape[0]) * np.dot(np.transpose(X), softmax_scores) # [X * (S - 1)] / N
      # calculate the partial derivative of L2 regularization wrt W (lambda * 2 * W)
      dW_l1 = reg * (np.abs(W) / W)
      # Sum all of the gradients(dL2/dW + dsoftmax/dW)
      dW += dW_l1


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
