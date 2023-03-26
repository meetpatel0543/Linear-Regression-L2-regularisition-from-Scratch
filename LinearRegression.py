import numpy as np

class LinearRegression:
  def __init__(self, batch_size=32, regularization=0, max_epochs=100, patience=3):
    """Linear Regression using Gradient Descent.
    Parameters:
    -----------
    batch_size: int
        The number of samples per batch.
    regularization: float
        The regularization parameter.
    max_epochs: int
        The maximum number of epochs.
    patience: int
        The number of epochs to wait before stopping if the validation loss
        does not improve.
    """
    self.batch_size = batch_size
    self.regularization = regularization
    self.max_epochs = max_epochs
    self.patience = patience
    self.weights = None
    self.bias = None

  def fit(self, X, y, batch_size=32, regularization=0, max_epochs=100, patience=3, alpha= 0.01):
    """Fit a linear model.
    Parameters:
    -----------
    batch_size: int
        The number of samples per batch.
    regularization: float
        The regularization parameter.
    max_epochs: int
        The maximum number of epochs.
    patience: int
        The number of epochs to wait before stopping if the validation loss
        does not improve.
    """
    self.batch_size = batch_size
    self.regularization = regularization
    self.max_epochs = max_epochs
    self.patience = patience
    self.train_error_list = []
    self.validation_error_list = []

    np.random.seed(10)
    # TODO: Initialize the weights and bias based on the shape of X and y.
    # add 1 in input for bias term
    X = np.hstack((np.ones((X.shape[0],1)),X))
    n_samples, n_features = X.shape

    # split X into train and validation
    X_train_2 = X[:int(0.9*n_samples),:]
    Y_train_2 = y[:int(0.9*n_samples)]
    X_validation = X[int(0.9*n_samples):,:]
    Y_validation = y[int(0.9*n_samples):]
    
    self.weights = np.random.uniform(-1,1,(n_features,))

    def gradient(X,y):
      return (2/len(y))*np.sum(np.dot(X,self.weights)-y)

    def l2_gradient(X,y):
      return (1/len(y))* np.dot(X.T,(np.dot(X,self.weights)-y))+(2*self.regularization*self.weights)

    def l2_regularization(X,y):
      return (1/len(y))*np.sum((np.dot(X,self.weights)-y)**2)

    def mean_squared(X,target):
      return np.sum((np.dot(X,self.weights)-target)**2)/len(target)

    # TODO: Implement the training loop.
    print("epoch--------->training_loss------------->validation_loss")
    for epoch in range(self.max_epochs):
      number_of_batches = X_train_2.shape[0]//self.batch_size
      i = 0
      error = 0
      for i in range(number_of_batches+1):
        input = X_train_2[i * batch_size:(i + 1)*batch_size, :]
        target = Y_train_2[i * batch_size:(i + 1)*batch_size]
        if self.regularization == 0:
          self.weights = self.weights - alpha * gradient(input,target)
          error += mean_squared(input,target)
        else:
          self.weights = self.weights - alpha * l2_gradient(input,target)
          error += l2_regularization(input,target)
        
      # if any samples are remaining to be iterated
      if X_train_2.shape[0]%batch_size != 0:
        input = X_train_2[i * batch_size:X.shape[0]]
        target = Y_train_2[i * batch_size:X.shape[0]]
        if self.regularization == 0:
          self.weights = self.weights - alpha * gradient(input,target)
          error += mean_squared(input,target)
        else:
          self.weights = self.weights - alpha * l2_gradient(input,target)
          error += l2_regularization(input,target)
        i = i+1
      
      self.train_error_list.append(error)
      # validation test score
      val_loss = mean_squared(X_validation,Y_validation)
      self.validation_error_list.append(val_loss)

      print(epoch,"--------->",error,"----------->",val_loss)
      # check Patience/Early stopping
      if len(self.validation_error_list)>patience:
        last_n_elements = self.validation_error_list[-self.patience:]
        if last_n_elements==sorted(last_n_elements):
          print("-------------------Early stopping as Patience criteria reached----------------------")
          break

  def predict(self, X):
    """Predict using the linear model.
    Parameters
    ----------
    X: numpy.ndarray
        The input data.
    """
    # TODO: Implement the prediction function.
    try:
        return np.dot(np.hstack((np.ones((X.shape[0],1)),X[:,None])),self.weights)
    except:
        return np.dot(np.hstack((np.ones((X.shape[0],1)),X)),self.weights)
  def score(self, X, y):
    """Evaluate the linear model using the mean squared error.
    Parameters
    ----------
    X: numpy.ndarray
        The input data.
    y: numpy.ndarray
        The target data.
    """
    # TODO: Implement the scoring function.
    return np.sum((X-y)**2)/len(y)
    
