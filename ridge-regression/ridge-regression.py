
# coding: utf-8

# # Ridge regression (interpretation)

# ### Fire up graphlab create

# In[1]:

import graphlab


# ### Polynomial regression function

# In[60]:

#Create an SFrame consisting of the powers of an SArray up to a specific degree:
def polynomial_sframe(feature, degree):
    
    poly_sframe = graphlab.SFrame() # creates an empty sframe
    
    poly_sframe['power_1'] = feature # set the feature sarray as the first column of the sframe
  
    if degree > 1: # first check if degree > 1
        for power in range(2, degree + 1): 
            
            name = 'power_' + str(power) # name the column according to the degree
            
            poly_sframe[name] = feature.apply(lambda x : x**power ) #assign poly_sframe[name] to the appropriate power of feature

    return poly_sframe    


# Let's use matplotlib to visualize what a polynomial regression looks like on the house data.

# In[4]:

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[5]:

sales = graphlab.SFrame('kc_house_data.gl/')


# For plotting purposes (connecting the dots), we need to sort by the values of sqft_living. For houses with identical square footage, we break the tie by their prices.

# In[61]:

sales = sales.sort(['sqft_living','price'])


# Let us revisit the 15th-order polynomial model using the 'sqft_living' input. Generate polynomial features up to degree 15 using `polynomial_sframe()` and fit a model with these features. When fitting the model, use an L2 penalty of `1e-5`:

# In[62]:

l2_small_penalty = 1e-5


# Step 1: Create 15 degree polynomial sframe of sqft_living feature of the dataset, copy the corresponding price column, copy all feaures of the polynomial sframe to an sarray

# In[63]:

poly_data = polynomial_sframe(sales['sqft_living'], 15)
poly_features = poly_data.column_names()
poly_data['price'] = sales['price']


# Step 2: Build model for the dataset

# In[79]:

model = graphlab.linear_regression.create(poly_data, target='price', features=['power_1'], validation_set=None, 
                                             l2_penalty=l2_small_penalty, verbose=False)


# In[80]:

model.get('coefficients').print_rows(num_rows=16)


# ### Observe overfitting

# First, split the data into split the sales data into four subsets of roughly equal size and call them `set_1`, `set_2`, `set_3`, and `set_4`. Use `.random_split` function and make sure you set `seed=0`. 

# In[68]:

(semi_split1, semi_split2) = sales.random_split(.5,seed=0)
(set_1, set_2) = semi_split1.random_split(0.5, seed=0)
(set_3, set_4) = semi_split2.random_split(0.5, seed=0)


# Step 1: Polynomial sframes for all the 4 sets

# In[16]:

set_1_poly_data = polynomial_sframe(set_1['sqft_living'], 15)
set_2_poly_data = polynomial_sframe(set_2['sqft_living'], 15)
set_3_poly_data = polynomial_sframe(set_3['sqft_living'], 15)
set_4_poly_data = polynomial_sframe(set_4['sqft_living'], 15)


# Step 2: Adding the corresponding price column to all the 4 sets

# In[69]:

set_1_poly_data['price'] = set_1['price']
set_2_poly_data['price'] = set_2['price']
set_3_poly_data['price'] = set_3['price']
set_4_poly_data['price'] = set_4['price']


# Step 3: Building models for all the 4 sets

# In[70]:

model1 = graphlab.linear_regression.create(set_1_poly_data, target='price', features=poly_features, validation_set=None, 
                                           l2_penalty=l2_small_penalty, verbose=None)
model2 = graphlab.linear_regression.create(set_2_poly_data, target='price', features=poly_features, validation_set=None, 
                                           l2_penalty=l2_small_penalty, verbose=None)
model3 = graphlab.linear_regression.create(set_3_poly_data, target='price', features=poly_features, validation_set=None, 
                                           l2_penalty=l2_small_penalty, verbose=None)
model4 = graphlab.linear_regression.create(set_4_poly_data, target='price', features=poly_features, validation_set=None, 
                                           l2_penalty=l2_small_penalty, verbose=None)


# Step 4: Plotting the models of all the four sets

# In[71]:

#Model 1
plt.plot(set_1_poly_data['power_1'], set_1_poly_data['price'], '.',
        set_1_poly_data['power_1'], model1.predict(set_1_poly_data), '-')


# In[72]:

#Model 2
plt.plot(set_2_poly_data['power_1'], set_2_poly_data['price'], '.',
        set_2_poly_data['power_1'], model2.predict(set_2_poly_data), '-')


# In[73]:

#Model 3
plt.plot(set_3_poly_data['power_1'], set_3_poly_data['price'], '.',
        set_3_poly_data['power_1'], model3.predict(set_3_poly_data), '-')


# In[74]:

#Model 4
plt.plot(set_4_poly_data['power_1'], set_4_poly_data['price'], '.',
        set_4_poly_data['power_1'], model4.predict(set_4_poly_data), '-')


# The model curves differ greatly in this models with lower l2 penalty

# Step 5: Getting the coefficients of each model of the 4 sets

# In[75]:

model1.get('coefficients')


# In[76]:

model2.get('coefficients')


# In[77]:

model3.get('coefficients')


# In[78]:

model4.get('coefficients')


# ### So it can be seen that the value of weights changes with the variation in data as first we used the full dataset to make our model and then we used 4 different subsets of our dataset to make 4 different models
# ### Ridge regression now comes to the rescue

# Generally, whenever we see weights change so much in response to change in data, we believe the variance of our estimate to be large. Ridge regression aims to address this issue by penalizing "large" weights.

# ### New L2 penalty (larger)

# In[81]:

l2_penalty = 1e5


# In[82]:

new_model1 = graphlab.linear_regression.create(set_1_poly_data, target='price', features=poly_features, validation_set=None, 
                                           l2_penalty=l2_penalty, verbose=None)
new_model2 = graphlab.linear_regression.create(set_2_poly_data, target='price', features=poly_features, validation_set=None, 
                                           l2_penalty=l2_penalty, verbose=None)
new_model3 = graphlab.linear_regression.create(set_3_poly_data, target='price', features=poly_features, validation_set=None, 
                                           l2_penalty=l2_penalty, verbose=None)
new_model4 = graphlab.linear_regression.create(set_4_poly_data, target='price', features=poly_features, validation_set=None, 
                                           l2_penalty=l2_penalty, verbose=None)


# Print coefficients of new models

# In[83]:

new_model1.get('coefficients')


# In[84]:

new_model2.get('coefficients')


# In[85]:

new_model3.get('coefficients')


# In[86]:

new_model4.get('coefficients')


# Plotting the models of all the 4 sets

# In[88]:

#New Model 1
plt.plot(set_1_poly_data['power_1'], set_1_poly_data['price'], '.', 
         set_1_poly_data['power_1'], new_model1.predict(set_1_poly_data), '-')


# In[89]:

#New Model 2
plt.plot(set_2_poly_data['power_1'], set_2_poly_data['price'], '.', 
         set_2_poly_data['power_1'], new_model2.predict(set_2_poly_data), '-')


# In[90]:

#New Model 3
plt.plot(set_3_poly_data['power_1'], set_3_poly_data['price'], '.', 
         set_3_poly_data['power_1'], new_model3.predict(set_3_poly_data), '-')


# In[91]:

#New Model 4
plt.plot(set_4_poly_data['power_1'], set_4_poly_data['price'], '.', 
         set_4_poly_data['power_1'], new_model4.predict(set_4_poly_data), '-')


# These curves should vary a lot less, now that you applied a high degree of regularization.

# ### In these new models with higher l2 penalty we can see the weights differ less and the corresponding model predictions also have less variation in the curves

# ### Selecting an L2 penalty via K folds cross-validation

# Set aside segment k-1 as the validation set, and fit a model on rest of data, and evalutate it on this validation set
# 
# After this process, we compute the average of the k validation errors, and use it as an estimate of the generalization error. Notice that  all observations are used for both training and validation, as we iterate over segments of data. 
# 
# To estimate the generalization error well, it is crucial to shuffle the training data before dividing them into segments.

# In[93]:

(train_valid, test) = sales.random_split(.9, seed=1)
train_valid_shuffled = graphlab.toolkits.cross_validation.shuffle(train_valid, random_seed=1)


# Once the data is shuffled, we divide it into equal segments. Each segment should receive `n/k` elements, where `n` is the number of observations in the training set and `k` is the number of segments. Since the segment 0 starts at index 0 and contains `n/k` elements, it ends at index `(n/k)-1`. The segment 1 starts where the segment 0 left off, at index `(n/k)`. With `n/k` elements, the segment 1 ends at index `(n*2/k)-1`. Continuing in this fashion, we deduce that the segment `i` starts at index `(n*i/k)` and ends at `(n*(i+1)/k)-1`.

# With this pattern in mind, we write a short loop that prints the starting and ending indices of each segment, just to make sure you are getting the splits right.

# In[97]:

n = len(train_valid_shuffled)
k = 10 # 10-fold cross-validation
print "Dataset length: " +str(n)
print "No. of segments: " +str(k)
print "Segments:"
for i in xrange(k):
    start = (n*i)/k
    end = (n*(i+1))/k-1
    print i, (start, end)


# In[98]:

train_valid_shuffled[0:10] # rows 0 to 9


# Now let us extract individual segments with array slicing. Consider the scenario where we group the houses in the `train_valid_shuffled` dataframe into k=10 segments of roughly equal size, with starting and ending indices computed as above.
# Extract the fourth segment (segment 3) and assign it to a variable called `validation4`.

# In[99]:

validation4 = train_valid_shuffled[5818:7758]


# Extract the remainder of the data after *excluding* fourth segment (segment 3) and assign the subset to `train4`.

# In[103]:

train4 = train_valid_shuffled[0:5818].append(train_valid_shuffled[7758:])


# Now we are ready to implement k-fold cross-validation. Write a function that computes k validation errors by designating each of the k segments as the validation set. It accepts as parameters (i) `k`, (ii) `l2_penalty`, (iii) dataframe, (iv) name of output column (e.g. `price`) and (v) list of feature names. The function returns the average validation error using k segments as validation sets.
# 
# * For each i in [0, 1, ..., k-1]:
#   * Compute starting and ending indices of segment i and call 'start' and 'end'
#   * Form validation set by taking a slice (start:end+1) from the data.
#   * Form training set by appending slice (end+1:n) to the end of slice (0:start).
#   * Train a linear model using training set just formed, with a given l2_penalty
#   * Compute validation error using validation set just formed

# ### K-folds cross validation algorithm

# In[174]:

def k_fold_cross_validation(k, l2_penalty, data, output_name, features_list):
    n = len(data)
    validation_error_sum = 0
    for i in xrange(k):
        start = (n*i)/k
        end = (n*(i+1))/k-1
        validation_set = data[start:end+1]
        train_set = data[0:start].append(data[end+1:n])
        linear_model = graphlab.linear_regression.create(train_set, target=output_name, features=features_list, 
                                                         validation_set=None, l2_penalty = l2_penalty, verbose=False)
        model_predictions = linear_model.predict(validation_set)
        residuals = model_predictions - validation_set[output_name]
        residuals_squared = residuals*residuals
        residuals_sums_squared = residuals_squared.sum()
        validation_error_sum = validation_error_sum + residuals_sums_squared
    avg_validation_error = validation_error_sum/k
    return (avg_validation_error)


# Once we have a function to compute the average validation error for a model, we can write a loop to find the model that minimizes the average validation error. Write a loop that does the following:
# * We will again be aiming to fit a 15th-order polynomial model using the `sqft_living` input
# * For `l2_penalty` in [10^1, 10^1.5, 10^2, 10^2.5, ..., 10^7] (to get this in Python, you can use this Numpy function: `np.logspace(1, 7, num=13)`.)
#     * Run 10-fold cross-validation with `l2_penalty`
# * Report which L2 penalty produced the lowest average validation error.
# 
# Note: since the degree of the polynomial is now fixed to 15, to make things faster, you should generate polynomial features in advance and re-use them throughout the loop. Make sure to use `train_valid_shuffled` when generating polynomial features!

# In[175]:

import numpy as ng

#New l2 penalties
l2_penalties = ng.logspace(1, 7, num=13)
print l2_penalties


# In[176]:

#New polynomial sframe from the shuffled data, getting its features and appending price column
new_poly_sframe = polynomial_sframe(train_valid_shuffled['sqft_living'], 15)
new_features = new_poly_sframe.column_names()
new_poly_sframe['price'] = train_valid_shuffled['price']


# In[177]:

#Data exploration
new_poly_sframe[0:1]


# In[178]:

k = 10
l2_with_error = {}
for i in xrange(13):
    error = k_fold_cross_validation(k, l2_penalties[i], new_poly_sframe, 'price', new_features)
    l2_with_error[l2_penalties[i]] = error


# ***The best value for the L2 penalty according to 10-fold validation***

# In[179]:

best_l2_penalty = min(l2_with_error, key=l2_with_error.get)
print best_l2_penalty


# In[180]:

# Plot the l2_penalty values in the x axis and the cross-validation error in the y axis.
x = []
y = []
for key, value in l2_with_error.iteritems():
    x.append(key)
    y.append(value)
plt.plot(x, y)
plt.xscale('log')
plt.xlabel("L2 Penalty")
plt.ylabel("Validation Error")


# ### Final model on the entire dataset

# In[181]:

#Make a polynomial sframe of the training dataset previously defined
training_sframe = polynomial_sframe(train_valid['sqft_living'], 15)
training_features = training_sframe.column_names()
training_sframe['price'] = train_valid['price']


# In[182]:

#Build Model
final_model = graphlab.linear_regression.create(training_sframe, target='price', features=training_features, 
                                                validation_set=None, l2_penalty=best_l2_penalty)


# In[183]:

#Print coefficients
final_model.get('coefficients').print_rows(num_rows=16)


# In[184]:

#Plot model predictions and actual predictions for power_1
plt.plot(training_sframe['power_1'], training_sframe['price'], '.', 
        training_sframe['power_1'], final_model.predict(training_sframe), '-')


# ### RSS on the TEST data using the final model learned with best L2 penalty

# In[185]:

#Creating sframe for the test dataset
test_sframe = polynomial_sframe(test['sqft_living'], 15)
test_sframe['price'] = test['price']

#Calculating the RSS using final model on the test dataset
final_model_predictions = final_model.predict(test_sframe)
actual_predictions = test_sframe['price']
res = actual_predictions - final_model_predictions
res_squared = res*res
res_errors_sum_squared = res_squared.sum()
print res_errors_sum_squared

