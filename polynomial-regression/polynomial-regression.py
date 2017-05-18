
# coding: utf-8

# # Assessing Fit (polynomial regression)

# ### Fire up graphlab create

# In[1]:

import graphlab


# ### Polynomial sframe function

# In[6]:

#Create an SFrame consisting of the powers of an SArray up to a specific degree:
def polynomial_sframe(feature, degree):
    
    poly_sframe = graphlab.SFrame() # creates an empty sframe
    
    poly_sframe['power_1'] = feature # set the feature sarray as the first column of the sframe
  
    if degree > 1: # first check if degree > 1
        for power in range(2, degree + 1): 
            
            name = 'power_' + str(power) # name the column according to the degree
            
            poly_sframe[name] = feature.apply(lambda x : x**power ) #assign poly_sframe[name] to the appropriate power of feature

    return poly_sframe


# Test the function

# In[7]:

tmp = graphlab.SArray([1., 2., 3.])
print polynomial_sframe(tmp, 3)


# ### Visualizing polynomial regression

# Let's use matplotlib to visualize what a polynomial regression looks like on some real data.

# In[8]:

sales = graphlab.SFrame('kc_house_data.gl/')


# ### Data exploration

# In[11]:

sales[0:1]


# Going to use the **sqft_living** variable. For plotting purposes (connecting the dots), we need to sort by the values of sqft_living. For houses with identical square footage, we break the tie by their prices.

# In[9]:

sales = sales.sort(['sqft_living', 'price'])


# ### Degree 1 polynomial using 'sqft_living' (i.e. a line) to predict 'price' and plot what it looks like.

# In[13]:

poly1_data = polynomial_sframe(sales['sqft_living'], 1)
poly1_data['price'] = sales['price'] # add price to the data since it's the target


# In[14]:

poly1_data


# In[15]:

model1 = graphlab.linear_regression.create(poly1_data, target = 'price', features = ['power_1'], validation_set = None)


# In[16]:

model1.get("coefficients")


# In[17]:

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[18]:

plt.plot(poly1_data['power_1'],poly1_data['price'],'.',
        poly1_data['power_1'], model1.predict(poly1_data),'-')


# The first pair of SArrays we passed are the 1st power of sqft and the actual price we then ask it to print these as dots '.'. The next pair we pass is the 1st power of sqft and the predicted values from the linear model. We ask these to be plotted as a line '-'. 
# 
# We can see, not surprisingly, that the predicted values all fall on a line, specifically the one with slope 280 and intercept -43579. What if we wanted to plot a second degree polynomial?

# ### Degreee two polynomial

# In[19]:

poly2_data = polynomial_sframe(sales['sqft_living'], 2)
my_features = poly2_data.column_names() # get the name of the features
poly2_data['price'] = sales['price'] # add price to the data since it's the target
model2 = graphlab.linear_regression.create(poly2_data, target = 'price', features = my_features, validation_set = None)


# In[20]:

model2.get("coefficients")


# In[23]:

plt.plot(poly2_data['power_1'],poly2_data['price'],'.',
        poly2_data['power_1'], model2.predict(poly2_data),'-')


# The resulting model looks like half a parabola. Try on your own to see what the cubic looks like:

# ### Degree three polynomial

# In[24]:

poly3_data = polynomial_sframe(sales['sqft_living'], 3)
poly3_features = poly3_data.column_names()
poly3_data['price'] = sales['price']
model3 = graphlab.linear_regression.create(poly3_data, target = 'price', features = poly3_features, validation_set = None)


# In[25]:

model3.get('coefficients')


# In[26]:

plt.plot(poly3_data['power_1'],poly3_data['price'],'.',
        poly3_data['power_1'], model3.predict(poly3_data),'-')


# ### 15th degree polynomial

# In[27]:

poly15_data = polynomial_sframe(sales['sqft_living'], 15)
poly15_features = poly15_data.column_names()
poly15_data['price'] = sales['price']
model4 = graphlab.linear_regression.create(poly15_data, target = 'price', features = poly15_features, validation_set = None)


# In[29]:

model4.get("coefficients")


# In[30]:

plt.plot(poly15_data['power_1'],poly15_data['price'],'.',
        poly15_data['power_1'], model4.predict(poly15_data),'-')


# ### Changing the data and re-learning

# Split the sales data into four subsets of roughly equal size. Then estimate a 15th degree polynomial model on all four subsets of the data.

# In[35]:

(set_A, set_B) = sales.random_split(0.5, seed=0)


# In[36]:

(set_1, set_2) = set_A.random_split(0.5, seed=0)


# In[37]:

(set_3, set_4) = set_B.random_split(0.5, seed=0)


# ### Fitting a 15th degree polynomial on set_1, set_2, set_3, and set_4 using sqft_living to predict prices.

# In[39]:

#set_1
poly_set_1 = polynomial_sframe(set_1['sqft_living'], 15)
set_1_features = poly_set_1.column_names()
poly_set_1['price'] = set_1['price']
model_set_1 = graphlab.linear_regression.create(poly_set_1, target = 'price', features = set_1_features, validation_set = None)


# In[42]:

model_set_1.get('coefficients').print_rows(num_rows=16)


# In[41]:

plt.plot(poly_set_1['power_1'], poly_set_1['price'],'.',
        poly_set_1['power_1'], model_set_1.predict(poly_set_1),'-')


# In[43]:

#set_2
poly_set_2 = polynomial_sframe(set_2['sqft_living'], 15)
set_2_features = poly_set_2.column_names()
poly_set_2['price'] = set_2['price']
model_set_2 = graphlab.linear_regression.create(poly_set_2, target = 'price', features = set_2_features, validation_set = None)


# In[44]:

model_set_2.get('coefficients').print_rows(num_rows=16)


# In[45]:

plt.plot(poly_set_2['power_1'], poly_set_2['price'],'.',
        poly_set_2['power_1'], model_set_2.predict(poly_set_2),'-')


# In[46]:

#set_3
poly_set_3 = polynomial_sframe(set_3['sqft_living'], 15)
set_3_features = poly_set_3.column_names()
poly_set_3['price'] = set_3['price']
model_set_3 = graphlab.linear_regression.create(poly_set_3, target = 'price', features = set_3_features, validation_set = None)


# In[47]:

model_set_3.get('coefficients').print_rows(num_rows=16)


# In[48]:

plt.plot(poly_set_3['power_1'], poly_set_3['price'],'.',
        poly_set_3['power_1'], model_set_3.predict(poly_set_3),'-')


# In[49]:

#set_4
poly_set_4 = polynomial_sframe(set_4['sqft_living'], 15)
set_4_features = poly_set_4.column_names()
poly_set_4['price'] = set_4['price']
model_set_4 = graphlab.linear_regression.create(poly_set_4, target = 'price', features = set_4_features, validation_set = None)


# In[50]:

model_set_4.get('coefficients').print_rows(num_rows=16)


# In[51]:

plt.plot(poly_set_4['power_1'], poly_set_4['price'], '.',
        poly_set_4['power_1'], model_set_4.predict(poly_set_4), '-')


# ### Selecting a Polynomial Degree

# Split the sales dataset 3-way into training set, test set, and validation

# In[52]:

(training_and_validation, testing) = sales.random_split(0.9, seed=1)
(training, validation) = training_and_validation.random_split(0.5, seed=1)


# Next you should write a loop that does the following:
# * For degree in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] (to get this in python type range(1, 15+1))
#     * Build an SFrame of polynomial data of train_data['sqft_living'] at the current degree
#     * hint: my_features = poly_data.column_names() gives you a list e.g. ['power_1', 'power_2', 'power_3'] which you might find useful for graphlab.linear_regression.create( features = my_features)
#     * Add train_data['price'] to the polynomial SFrame
#     * Learn a polynomial regression model to sqft vs price with that degree on TRAIN data
#     * Compute the RSS on VALIDATION data (here you will want to use .predict()) for that degree and you will need to make a polynmial SFrame using validation data.
# * Report which degree had the lowest RSS on validation data (remember python indexes from 0)
# 
# (Note you can turn off the print out of linear_regression.create() with verbose = False)

# In[119]:

def rss_by_degree(data):

    # initialize the degree
    degree = 1

    # initialize the dictionary
    degree_rss = {}

    for degree in range(1, 16):

        # built a polynomial sframe
        poly_data = polynomial_sframe(training['sqft_living'], degree)

        # get the features
        poly_features = poly_data.column_names()

        # copy the target
        poly_data['price'] = training['price']

        # build the model
        poly_model = graphlab.linear_regression.create(poly_data, target='price', features=poly_features, validation_set=None, 
                                                       verbose=False)
        # calculate RSS
        residual_list = poly_model.predict(polynomial_sframe(data['sqft_living'], degree)) - data['price']
        residual_squares = residual_list * residual_list
        residual_sum_of_squares = residual_squares.sum()

        # add degree and RSS to dictionary
        degree_rss[degree] = residual_sum_of_squares
        
    return degree_rss


# ### Run function on validation set

# In[123]:

validation_rss = rss_by_degree(validation)


# **Which degree (1, 2, …, 15) had the lowest RSS on Validation data?**

# In[124]:

validation_rss


# In[125]:

min(validation_rss, key=validation_rss.get)


# ### Run function on test set

# In[126]:

test_rss = rss_by_degree(testing)


# **Which degree (1, 2, …, 15) had the lowest RSS on Test data?**

# In[127]:

test_rss


# In[128]:

min(test_rss, key=test_rss.get)


# **What is the RSS on TEST data for the model with the degree selected from Validation data?**

# In[129]:

test_rss.get(6)

