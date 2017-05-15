
# coding: utf-8

# # Multiple Regression (gradient descent) with Numpy

# ### Fire up graphlab create

# In[1]:

import graphlab


# ### Load in house sales data
# 
# Dataset is from house sales in King County, the region where the city of Seattle, WA is located.

# In[3]:

sales = graphlab.SFrame('kc_house_data.gl/')


# ### Data exploration

# In[6]:

sales[0:1]


# ### Convert SFrame to Numpy array

# In[4]:

import numpy as np


# In[16]:

# function to convert sframe to numpy array (matrix)
def get_numpy_data(data_sframe, features, output):
    
    data_sframe['constant'] = 1 # new constant column in the sframe signifying intercept
    
    features = ['constant'] + features # prepend constant to features list
    
    features_sframe = data_sframe[features] # new sframe selecting columns from data_sframe mentioned in features list

    feature_matrix = features_sframe.to_numpy() # convert sframe to numpy matrix

    output_sarray = data_sframe['price'] # an sarray consisting of the output column

    output_array = output_sarray.to_numpy() # converts sarray to a numpy array

    return(feature_matrix, output_array)


# ### Test the function 

# In[20]:

(example_features, example_output) = get_numpy_data(sales, ['sqft_living'], 'price')
print example_features[0:1] # the first row of the data
print example_output[0:1] # and the corresponding output


# ### Predicting output given regression weights

# Suppose we had the weights [1.0, 1.0] and the features [1.0, 1180.0] and we wanted to compute the predicted output 1.0\*1.0 + 1.0\*1180.0 = 1181.0 this is the dot product between these two arrays. If they're numpy arrayws we can use np.dot() to compute this:

# In[21]:

my_weights = np.array([1., 1.]) # example weights
my_features = example_features[0,] # first data point
predicted_value = np.dot(my_features, my_weights)
print predicted_value


# ### Function to predict output given feature matrix and weight vector

# In[22]:

def predict_output(feature_matrix, weights):
    predictions = np.dot(feature_matrix, weights)
    return(predictions)


# ### Test the function

# In[23]:

test_predictions = predict_output(example_features, my_weights)
print test_predictions[0] # should be 1181.0
print test_predictions[1] # should be 2571.0


# ### Computing the Derivative

# RSS (error) for 1 data point is:
# 
# (w[0]\*[CONSTANT] + w[1]\*[feature_1] + ... + w[i] \*[feature_i] + ... +  w[k]\*[feature_k] - output)^2
# 
# So the derivative with respect to weight w[i] by the chain rule is:
# 
# 2\*(w[0]\*[CONSTANT] + w[1]\*[feature_1] + ... + w[i] \*[feature_i] + ... +  w[k]\*[feature_k] - output)\* [feature_i]
# 
# In short:
# 
# 2\*error\*[feature_i]
# 
# That is, the derivative for the weight for feature i is the sum (over data points) of 2 times the product of the error and the feature itself. In the case of the constant then this is just twice the sum of the errors!

# In[27]:

def feature_derivative(errors, feature):
    
    # Assume that errors and feature are both numpy arrays of the same length (number of data points)
    dot_product = np.dot(errors, feature)
    
    # compute twice the dot product of these vectors as 'derivative' and return the value
    derivative = 2 * dot_product

    return(derivative)


# ### Test function

# In[29]:

(example_features, example_output) = get_numpy_data(sales, ['sqft_living'], 'price') 
my_weights = np.array([0., 0.]) # this makes all the predictions 0
test_predictions = predict_output(example_features, my_weights) 
errors = test_predictions - example_output # prediction errors in this case is just the -example_output
feature = example_features[:,0] # let's compute the derivative with respect to 'constant', the ":" indicates "all rows"
derivative = feature_derivative(errors, feature)
print derivative
print -np.sum(example_output)*2 # should be the same as derivative


# ### Gradient Descent

# Here is a function that performs a gradient descent. Given a starting point we update the current weights by moving in the negative gradient direction. The gradient is the direction of *increase* and therefore the negative gradient is the direction of *decrease* and we're trying to *minimize* a cost function. 
# 
# The amount by which we move in the negative gradient *direction*  is called the 'step size'. We stop when we are 'sufficiently close' to the optimum. We define this by requiring that the magnitude (length) of the gradient vector to be smaller than a fixed 'tolerance'.

# In[30]:

from math import sqrt # the magnitude/length of a vector [g[0], g[1], g[2]] is sqrt(g[0]^2 + g[1]^2 + g[2]^2)


# In[33]:

def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):
    converged = False 
    weights = np.array(initial_weights) # converting to a numpy array
    
    while not converged:
        # compute the predictions based on feature_matrix and weights using your predict_output() function
        predictions = predict_output(feature_matrix, weights)
        
        # compute the errors as predictions - output
        errors = predictions - output

        gradient_sum_squares = 0 # initialize the gradient sum of squares
        
        # while we haven't reached the tolerance yet, update each feature's weight
        for i in range(len(weights)): # loop over each weight
 
            # compute the derivative for weight[i]:
            derivative_weight_i = feature_derivative(errors, feature_matrix[:, i])

            # add the squared value of the derivative to the gradient sum of squares (for assessing convergence)
            gradient_sum_squares = gradient_sum_squares + derivative_weight_i**2

            # subtract the step size times the derivative from the current weight
            weights[i] = weights[i] - (step_size * derivative_weight_i)
                
        # compute the square-root of the gradient sum of squares to get the gradient magnitude:
        gradient_magnitude = sqrt(gradient_sum_squares)
        if gradient_magnitude < tolerance:
            converged = True
    return(weights)


# Since the gradient is a sum over all the data points and involves a product of an error and a feature the gradient itself will be very large since the features are large (squarefeet) and the output is large (prices). So while you might expect "tolerance" to be small, small is only relative to the size of the features. 
# 
# For similar reasons the step size will be much smaller than you might expect but this is because the gradient has such large values.

# # Running the Gradient Descent as Simple Regression (Simple model)

# First let's split the data into training and test data.

# In[34]:

train_data,test_data = sales.random_split(.8,seed=0)


# In[36]:

simple_features = ['sqft_living']
my_output= 'price'
(simple_feature_matrix, output) = get_numpy_data(train_data, simple_features, my_output)
initial_weights = np.array([-47000., 1.])
step_size = 7e-12
tolerance = 2.5e7


# In[38]:

simple_weights = regression_gradient_descent(simple_feature_matrix, output,initial_weights, step_size, tolerance)
print simple_weights


# ### Get predictions for test data using new weights (Simple model)

# In[39]:

(test_simple_feature_matrix, test_output) = get_numpy_data(test_data, simple_features, my_output)


# In[41]:

simple_predictions = predict_output(test_simple_feature_matrix, simple_weights)
print simple_predictions


# **What is the predicted price for the 1st house in the TEST data set for model 1 (round to nearest dollar)?**

# In[74]:

simple_predictions[0]


# ### RSS function

# In[44]:

def RSS (predicted_output, true_output):
    difference = true_output - predicted_output
    squared_difference = difference * difference
    sum_of_squared_difference = squared_difference.sum()
    return (sum_of_squared_difference)


# In[56]:

output[5000]


# ### RSS for Simple model

# In[61]:

rss = RSS(simple_predictions, test_output)
print "Residual sum of squares error for Simple model: " +str(rss)


# # Running a multiple regression

# Now we will use more than one actual feature. Use the following code to produce the weights for a second model with the following parameters:

# In[62]:

model_features = ['sqft_living', 'sqft_living15'] # sqft_living15 is the average squarefeet for the nearest 15 neighbors. 
my_output = 'price'
(feature_matrix, output) = get_numpy_data(train_data, model_features, my_output)
initial_weights = np.array([-100000., 1., 1.])
step_size = 4e-12
tolerance = 1e9


# Use the above parameters to estimate the model weights. Record these values for your quiz.

# In[64]:

multiple_weights = regression_gradient_descent(feature_matrix, output,initial_weights, step_size, tolerance)
print multiple_weights


# ### Get predictions for test data using new weights (Multiple regression model)

# In[66]:

(test_multiple_feature_matrix, test_multiple_output) = get_numpy_data(test_data, model_features, my_output)


# In[68]:

multiple_predictions = predict_output(test_multiple_feature_matrix,  multiple_weights)
print multiple_predictions


# **What is the predicted price for the 1st house in the TEST data set for model 2?**

# In[73]:

multiple_predictions[0]


# **What is the actual price for the 1st house in the test data set?**

# In[72]:

test_multiple_output[0]


# # So the simple model is more closer to the actual price of the house 1

# RSS for Multiple regression model

# In[79]:

rss_multiple = RSS(multiple_predictions, test_multiple_output)
print "Residual sum of squares error for Multiple regression model: " +str(rss_multiple)


# # The multiple regression model has lower RSS than Simple model
