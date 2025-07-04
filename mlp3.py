# We implement the most basis mlp with a single hidden layer
# The code is based on a lecture by Prof. Th. Hofmann at ETH
import numpy as np
import scipy as sp


class mlp3():
    # Constructor
    # Input:    self = instance parameter
    #           n = width of input layer
    #           m = width of hidden layer
    #           method = loss function, either 'mse' or 'cross_entropy'
    #           seed = random seed for reproducibility
    #           lr = learning rate
    def __init__(self, n, m, method='mse', lr=0.01, seed=None):
        # set random seed for reproducibility
        self.seed = 42
        # if seed was passed as argument, update
        if seed is not None:
            self.seed = seed
        # define a module specific random number generator
        self.rng = np.random.default_rng(self.seed)
        # Initialize weight matrix with random entries
        self.W = self.rng.uniform(-1 / n, 1 / n, (m, n))   # mxn
        self.v = self.rng.uniform(-1 / m, 1 / m, (1, m))   # 1xm
        self.n = n  # width of input layer
        self.m = m  # width of hidden layer
        self.method = method  # loss function, either 'mse' or 'cross_entropy'
        self.lr = lr    # learning rate

    # Construct class from input data
    # Input:    cls = the class object
    #           x = a matrix of features, where each column is a sample
    #           m = width of hidden layer
    @classmethod
    def from_data(cls, x, m):
        # width of input layer equals number of rows of x
        n = x.shape[0]
        return cls(n, m)

    # Forward propagation as a method
    # Input:    self = instance parameter
    #           x = factor matrix where samples correspond to the s columns
    def forward(self, x):
        layers = {}
        # Compute pre-activation hidden layer
        pre_act_hidden = np.matmul(self.W, x)  # mxn . nxs = mxs
        # apply activation function (logit) to hidden layer
        hidden = sp.special.expit(pre_act_hidden)
        layers['Hidden'] = hidden
        pre_act_output = np.matmul(self.v, hidden)  # 1xm . mxs = 1xs
        if self.method == 'mse':
            # apply identity activation function to output layer
            layers['Output'] = pre_act_output
        elif self.method == 'cross_entropy':
            # apply sigmoid activation function to output layer
            layers['Output'] = sp.special.expit(pre_act_output)
        return layers

    # Compute the gradient for MSE or cross-entropy loss function
    # the cross-entropy loss arises in the binary classification problem, which
    # is modelled so that the output layer returns the probability of the
    # positive class.
    # The probability is obtained using the sigmoid for activation.
    # Denote by f(x) the pre-activation of the output layer and suppose that f
    # depends on a real parameter a (e.g., a coordinate of the output weight
    # vector v or a coordinate of the weight matrix W). We recall that using
    # the property sigma' = sigma * (1 - sigma), of the sigmoid functio and
    # letting delta^t be the difference between the activated output layer and
    # the true labels, we have that, letting f(x^t) the preactivation of the
    # output layer, for any parameter a
    #   dl / da = sum_t delta^t  * df / da (x^t)
    # We also recall that
    #   df / dv_i (x^t) = sigma(w_i x^t)
    #   df / dw_ij (x^t) = v_i * sensitivity_i^t * x_j^t
    # For MSE, one derives that
    #   (d l / d v)_i = 1 / s sum_t delta^t sigma(w_i x^t)
    #   (d l / d w)_ij = v_i / s sum_t delta^t * sensitivity_i^t * x_j^t
    # In particular, formally the gradients look the same, except for the value
    # of delta^t.
    # Input:    self = instance parameter
    #           x = a matrix of features, where each column is a sample,
    #               dimension = mxs
    #           y = a vector of labels of dimension 1xs
    def gradient(self, x, y):
        size = x.shape[1]  # number of samples
        layers = self.forward(x)
        hidden = layers['Hidden']   # mxs
        sv = hidden - hidden * hidden  # mxs - mxs * mxs = mxs
        out = layers['Output']      # 1xs
        delta = out - y             # 1xs
        # We compute the gradient for the output layer, i.e.,
        # delta * hidden is the 1xs * mxs = mxs array corresponding to
        #   delta^t * sigma(w_i x^t)
        # Then we sum over t and reshape to obtain the 1xm vector grad_v
        grad_v = np.sum(delta * hidden, axis=1).reshape(1, self.m) / size
        # We compute for every sample t the contribution to the gradient of the
        # weight-matrix, i.e.,
        # delta_t . v_i . sv_it . x_j^t
        # We first compute the mxs matrix containing the products v_i * sv_it
        sv_wt = self.v.T * sv  # mx1 * mxs = mxs
        # we compute delta_t * v_i * sv_it
        u = delta * sv_wt  # 1xs * mxs = mxs
        # Finally, we compute the gradient of the weight matrix
        grad_W = np.matmul(u, x.T) / size   # mxs . sxn = mxn
        g = {'v': grad_v, 'W': grad_W}
        return g

    # Create batches of indices for the traning data
    # Input:    self = instance parameter
    #           sample_size = number of samples in the training data
    #           batch_size = size of the batches, default is 32
    # Output:   a list of
    def batches(self, sample_size, batch_size=32):
        # If the batch size is larger than or equal to the sample size,
        # return all indices
        if batch_size > sample_size:
            return np.arange(sample_size).reshape(1, sample_size)
        else:
            # number of batches
            n_batches = sample_size // batch_size
            # Randomly select indices for the batch
            return self.rng.choice(
                sample_size,
                (n_batches, batch_size),
                replace=False)

    # Train the model using gradient descent by updating self.W and self.v
    # Input:    self = instance parameter
    #           x = sample features
    #           y = sample labels
    #           lr = learning rate, defaults to self.lr
    #           batch_size = batch size used for SGD
    #           epochs = number of epochs
    # Output:   None
    def optimize(self, x, y, lr=None, batch_size=32, epochs=100):
        if lr is None:
            lr = self.lr
        for epoch in range(epochs):
            # Get the batches of indices
            batches = self.batches(x.shape[1], batch_size)
            for batch in batches:
                # Get the current batch of data
                x_batch = x[:, batch]
                y_batch = y[batch]
                # Compute the gradient
                grad = self.gradient(x_batch, y_batch)
                # Update the weights
                self.W -= lr * grad['W']
                self.v -= lr * grad['v']

    # A method to reset the model
    # Input:    self = instance parameter
    def reset(self):
        # We just call the command used to initialize the weight matrices
        self.W = self.rng.uniform(-1 / self.n, 1 / self.n, (self.m, self.n))
        self.v = self.rng.uniform(-1 / self.m, 1 / self.m, (1, self.m))
