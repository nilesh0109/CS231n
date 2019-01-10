import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
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
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]

  full = np.exp(np.dot(X, W))
  full_sum = np.sum(full, axis=1)
  for i in range(num_train):
    fx = full[i] / full_sum[i]
    dW += np.dot(np.matrix(X[i]).T, np.matrix(fx))
    for j in range(num_classes):
      if j == y[i]:
        loss = loss - np.log(fx[j])

        dW[:, j] -= X[i]

  loss = loss / num_train + (reg * 0.5 * np.sum(np.square(W)))
  dW = dW / num_train + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
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
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  y_initial = np.exp(np.dot(X, W))
  y_total = np.sum(y_initial, axis=1)
  softmax = y_initial / np.matrix(y_total).T
  loss_correctClass = softmax[np.arange(num_train), y]
  loss_vector = -np.log(loss_correctClass)
  loss = np.mean(loss_vector) + reg * np.sum(W * W) / 2
  gradient_softmax = np.copy(softmax)
  gradient_softmax[np.arange(num_train), y] -= 1
  dW = np.dot(X.T, gradient_softmax) / num_train + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

