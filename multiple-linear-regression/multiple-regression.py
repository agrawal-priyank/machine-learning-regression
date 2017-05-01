
# coding: utf-8

# # Multiple Regression (Interpretation) on House sales data

# ### Fire up graphlab create

# In[1]:

import graphlab


# ### Load house sales data
# 
# Dataset is from house sales in King County, the region where the city of Seattle, WA is located.

# In[2]:

sales = graphlab.SFrame('kc_house_data.gl/')


# ### Data Exploration

# In[3]:

sales


# ### Split data into training and testing.
# 

# In[4]:

train_data,test_data = sales.random_split(.8,seed=0)


# ### Learning a multiple regression model

# ### Example Model

# In[5]:

example_features = ['sqft_living', 'bedrooms', 'bathrooms']
example_model = graphlab.linear_regression.create(train_data, target = 'price', features = example_features, 
                                                  validation_set = None)


# Extracting the regression weights (coefficients) as an SFrame as follows:

# In[6]:

example_weight_summary = example_model.get("coefficients")
print example_weight_summary


# ### Making Predictions

# In[7]:

example_predictions = example_model.predict(train_data)
print example_predictions[0]


# ### Compute RSS

# RSS function

# In[8]:

def get_residual_sum_of_squares(model, data, outcome):
    # First get the predictions
    predictions = model.predict(data)

    # Then compute the residuals/errors
    residuals = outcome - predictions

    # Then square and add them up
    residuals_squared = residuals*residuals
    RSS = residuals_squared.sum()

    return(RSS)    


# ### Run example model on test data

# In[12]:

rss_example_train = get_residual_sum_of_squares(example_model, test_data, test_data['price'])
print "The RSS for example model: " +str(rss_example_train)


# ### Create new features

# Transformations of existing features e.g. the log of the squarefeet or even "interaction" features such as the product of bedrooms and bathrooms.

# In[24]:

from math import log


# * bedrooms_squared = bedrooms\*bedrooms
# * bed_bath_rooms = bedrooms\*bathrooms
# * log_sqft_living = log(sqft_living)
# * lat_plus_long = lat + long 

# In[13]:

# create the bedroom squared new feature in both TEST and TRAIN data
train_data['bedrooms_squared'] = train_data['bedrooms'].apply(lambda x: x**2)
test_data['bedrooms_squared'] = test_data['bedrooms'].apply(lambda x: x**2)


# In[21]:

# create the bathroom bedroom new feature in both TEST and TRAIN data
train_data['bed_bath_rooms'] = train_data.apply(lambda x: x['bedrooms']*x['bathrooms'])
test_data['bed_bath_rooms'] = test_data.apply(lambda x: x['bedrooms']*x['bathrooms'])


# In[26]:

# create the log of sqft_living new feature in both TEST and TRAIN data
train_data['log_sqft_living'] = train_data['sqft_living'].apply(lambda x: log(x))
test_data['log_sqft_living'] = test_data['sqft_living'].apply(lambda x: log(x))


# In[28]:

# create the lat long new feature in both TEST and TRAIN data
train_data['lat_plus_long'] = train_data.apply(lambda x: x['lat'] + x['long'])
test_data['lat_plus_long'] = test_data.apply(lambda x: x['lat'] + x['long'])


# * Squaring bedrooms will increase the separation between not many bedrooms (e.g. 1) and lots of bedrooms (e.g. 4) since 1^2 = 1 but 4^2 = 16. Consequently this feature will mostly affect houses with many bedrooms.
# * bedrooms times bathrooms is an "interaction" feature. It is large when *both* of them are large.
# * Taking the log of squarefeet has the effect of bringing large values closer together and spreading out small values.
# * Adding latitude to longitude is totally non-sensical but we will see why later.

# In[31]:

# checking new features
train_data[0:1]


#  **What is the Mean (arithmetic average) values of 4 new features on TEST data?

# In[32]:

test_data['bedrooms_squared'].mean()


# In[33]:

test_data['bed_bath_rooms'].mean()


# In[34]:

test_data['log_sqft_living'].mean()


# In[35]:

test_data['lat_plus_long'].mean()


# ### Learning Multiple Models

# * Model 1: squarefeet, # bedrooms, # bathrooms, latitude & longitude
# * Model 2: add bedrooms\*bathrooms
# * Model 3: add log squarefeet, bedrooms squared, and the (nonsensical) latitude + longitude

# In[36]:

model_1_features = ['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long']
model_2_features = model_1_features + ['bed_bath_rooms']
model_3_features = model_2_features + ['bedrooms_squared', 'log_sqft_living', 'lat_plus_long']


# ### Model 1

# In[38]:

model_1 = graphlab.linear_regression.create(train_data, target = 'price', features = model_1_features, validation_set = None)
model_1_summary = model_1.get('coefficients')


# ### Model 2
# 

# In[40]:

model_2 = graphlab.linear_regression.create(train_data, target = 'price', features = model_2_features, validation_set = None)
model_2_summary = model_2.get('coefficients')


# ### Model 3

# In[41]:

model_3 = graphlab.linear_regression.create(train_data, target = 'price', features = model_3_features, validation_set = None)
model_3_summary = model_3.get('coefficients')


# ### Weights of each of the models

# In[42]:

model_1_summary


# In[43]:

model_2_summary


# In[44]:

model_3_summary


# ### Comparing multiple models
# 

# ### RSS of train data using all three models

# In[45]:

# Compute the RSS on TRAINING data for each of the three models and record the values:
rss_train_model_1 = get_residual_sum_of_squares(model_1, train_data, train_data['price'])
rss_train_model_2 = get_residual_sum_of_squares(model_2, train_data, train_data['price'])
rss_train_model_3 = get_residual_sum_of_squares(model_3, train_data, train_data['price'])


# **Which model (1, 2 or 3) has lowest RSS on TRAINING Data?**

# In[47]:

print "RSS of train data using model 1: " + str(rss_train_model_1)


# In[48]:

print "RSS of train data using model 2: " + str(rss_train_model_2)


# In[49]:

print "RSS of train data using model 3: " + str(rss_train_model_3)


# ### So RSS of the model 3 is lowest on the train data

# ### RSS of test data using all three models

# In[50]:

# Compute the RSS on TEST data for each of the three models and record the values:
rss_test_model_1 = get_residual_sum_of_squares(model_1, test_data, test_data['price'])
rss_test_model_2 = get_residual_sum_of_squares(model_2, test_data, test_data['price'])
rss_test_model_3 = get_residual_sum_of_squares(model_3, test_data, test_data['price'])


# **Which model (1, 2 or 3) has lowest RSS on TEST Data?**

# In[51]:

print "RSS of test data using model 1: " + str(rss_test_model_1)


# In[53]:

print "RSS of test data using model 2: " + str(rss_test_model_2)


# In[54]:

print "RSS of test data using model 3: " + str(rss_test_model_3)


# **Quiz Question: Which model (1, 2 or 3) has lowest RSS on TESTING Data?** Is this what you expected? Think about the features that were added to each model from the previous.

# ### So RSS of the model 2 is lowest on the test data

# # Hence model 2 is the optimal model
