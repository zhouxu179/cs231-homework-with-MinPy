# Writen by Xu Zhou, Sept 2016, Shanghaitech University, e-mail: zhouxu@shanghaitech.edu.cn

import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.
  
  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    
    ############################################################################
    # TODO: Initialize the weights and biases of the two-layer net. Weights    #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases using the keys 'W2' and 'b2'.                                 #
    ############################################################################
    W1 = weight_scale * np.random.randn(input_dim,hidden_dim)
    W2 = weight_scale * np.random.randn(hidden_dim,num_classes)
    b1 = np.zeros([1, hidden_dim])
    b2 = np.zeros([1, num_classes])
    self.params = {'W1': W1, 'W2': W2, 'b1': b1, 'b2':b2}
    
    
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
    # TODO: Implement the forward pass for the two-layer net, computing the    #
    # class scores for X and storing them in the scores variable.              #
    ############################################################################
    W1 = self.params['W1']
    W2 = self.params['W2']
    b1 = self.params['b1']
    b2 = self.params['b2']
    
    f1, cache1 = affine_relu_forward(X,W1,b1)
    scores, _ = affine_forward(f1,W2,b2)
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores

    
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the two-layer net. Store the loss  #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss using softmax, and make sure that grads[k] holds the gradients for  #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    reg = self.reg
    
    scores_max = np.max(scores, axis=1, keepdims=True)
    scores_exp = np.exp(scores - scores_max)
    scores_exp_sum = np.sum(scores_exp, axis=1, keepdims=True)
    scores_exp_n = scores_exp / scores_exp_sum
    N = scores.shape[0]
    ind = np.arange(N)
    loss_data = -np.sum(np.log(scores_exp_n[ind, y])) / N
    loss_reg = np.sum(W1**2) + np.sum(W2**2)
    
    dscore = scores_exp_n.copy()
    dscore[ind, y] -= 1
    dscore /= N
    
    df1, dW2, db2 = affine_backward(dscore, (f1,W2,b2))
    _, dW1, db1 = affine_relu_backward(df1, cache1)
    dW1 += reg*W1
    dW2 += reg*W2
    #db1 += reg*b1
    #db2 += reg*b2
    
    grads = {'W1':dW1,'W2':dW2,'b1':db1,'b2':db2}
    loss = loss_data + 0.5*reg*loss_reg
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################
    D = input_dim
    for i in xrange(self.num_layers):
      if i<self.num_layers-1:
        M = hidden_dims[i]
      else:
        M = num_classes
      W = weight_scale * np.random.randn(D,M)
      b = np.zeros([1, M])
      Wi = 'W{}'.format(i+1)
      bi = 'b{}'.format(i+1)
      self.params[Wi] = W
      self.params[bi] = b
      if i<self.num_layers-1 and self.use_batchnorm :
        gamma = np.random.randn(1,M)
        beta = b
        gammai = 'gamma{}'.format(i+1)
        betai = 'beta{}'.format(i+1) 
        self.params[gammai] = gamma
        self.params[betai] = beta
      D = M
      

    
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
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################n 
    fs = {'f0':X}
    caches = {}
    cachebns = {}
    cachedps = {}

    for i in xrange(self.num_layers):
      W = self.params['W{}'.format(i+1)]
      b = self.params['b{}'.format(i+1)]        
      if i < self.num_layers-1:
        if self.use_batchnorm:
          gamma = self.params['gamma{}'.format(i+1)]
          beta = self.params['beta{}'.format(i+1)]
          xaffine, cache = affine_forward(fs['f{}'.format(i)],W,b)
          bn_param = self.bn_params[i]
          xbn, cachebn = batchnorm_forward(xaffine, gamma, beta, bn_param)
          fi = np.maximum(xbn,0)       
          cachebns['cachebn{}'.format(i+1)] = cachebn        
        else:
          fi, cache = affine_relu_forward(fs['f{}'.format(i)],W,b)
        if self.use_dropout:
          fi, cachedp = dropout_forward(fi, self.dropout_param)
          cachedps['cachedp{}'.format(i+1)] = cachedp
      else:
        fi, cache = affine_forward(fs['f{}'.format(i)],W,b)
      caches['cache{}'.format(i+1)] = cache  
      fs['f{}'.format(i+1)] = fi
    
    scores = fi
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
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    reg = self.reg
    '''
    scores_max = np.max(scores, axis=1, keepdims=True)
    scores_shift = scores - scores_max
    scores_exp = np.exp(scores_shift)
    scores_exp_sum = np.sum(scores_exp, axis=1, keepdims=True)
    scores_exp_n = scores_exp / scores_exp_sum
    N = scores.shape[0]
    ind = np.arange(N)
    #loss_data = -np.sum(np.log(scores_exp_n[ind, y])) / N
    loss_data = np.sum(-scores_shift[ind,y] + np.log(scores_exp_sum)) / N
    '''
    
    scores_max = np.max(scores, axis=1, keepdims=True)
    scores_exp = np.exp(scores - scores_max)
    scores_exp_sum = np.sum(scores_exp, axis=1, keepdims=True)
    scores_exp_n = scores_exp / scores_exp_sum
    N = scores.shape[0]
    ind = np.arange(N)
    loss_data = -np.sum(np.log(scores_exp_n[ind, y])) / N

    
    
    
    dscore = scores_exp_n.copy()
    dscore[ind, y] -= 1
    dscore /= N
    
    W = self.params['W{}'.format(self.num_layers)]
    b = self.params['b{}'.format(self.num_layers)]
    df, dW, db = affine_backward(dscore, cache)
    dW += reg*W
    
    loss_reg = np.sum(W**2)
    
    grads['W{}'.format(self.num_layers)] = dW
    grads['b{}'.format(self.num_layers)] = db
    
    for i in range(self.num_layers-1,0,-1):
      W = self.params['W{}'.format(i)]
      loss_reg += np.sum(W**2)
      cache = caches['cache{}'.format(i)] 
      if self.use_dropout:
        cachedp = cachedps['cachedp{}'.format(i)]
        df = dropout_backward(df, cachedp)
      if self.use_batchnorm:
        cachebn = cachebns['cachebn{}'.format(i)]
        xbn = cachebn[-1]
        dout = df
        dout[xbn<0] = 0
        dx, dgamma, dbeta = batchnorm_backward(dout, cachebn)
        df, dW, db = affine_backward(dx, cache)   
        grads['gamma{}'.format(i)] = dgamma
        grads['beta{}'.format(i)] = dbeta
      else:  
        df, dW, db = affine_relu_backward(df, cache)
      dW += reg*W
      grads['W{}'.format(i)] = dW
      grads['b{}'.format(i)] = db   

    loss = loss_data + 0.5*reg*loss_reg
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads
