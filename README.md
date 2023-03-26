# Linear_Regression_L2_regularisition_from_Scratch.

Select 4 different combinations of input and output features to use to train 4 different models on iris dataset. For example, one model could predict the petal width given petal length and sepal width. Another could predict sepal length using only petal features. It does not matter which combination you choose as long as you have 4 unique combinations.

Your models should be trained using batch gradient descent with a batch size (optional parameter) of 32 using mean squared error as your loss function.

For each model, train for 
 steps (optional parameter) OR until the loss on the validation set increases for a number of consecutive epochs determined by patience (default to 3).

As each model trains, record the loss averaged over the batch size for each step. A single step is the processing of a single batch. One way to save this data is to either return an array from the fit method or save it as an internal class member that can be retrieved after training is complete.

After each model trains, plot the loss against the step number and save it. These plots should also be added to your report.

To observe the effects of regularization, pick one of your trained models and inspect the weights. Train an identical model again, except this time you will add L2 regularization to the loss. Record the difference in parameters between the regularized and non-regularized model.
