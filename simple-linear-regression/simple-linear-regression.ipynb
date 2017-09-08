{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Simple Linear Regression on House Sales data"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Fire up Graphlab Create"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "import graphlab"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Load in house sales data\n",
    "\n",
    "Dataset is from house sales in King County, the region where the city of Seattle, WA is located."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "sales = graphlab.SFrame('kc_house_data.gl/')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Explore house sales data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div style=\"max-height:1000px;max-width:1500px;overflow:auto;\"><table frame=\"box\" rules=\"cols\">\n",
       "    <tr>\n",
       "        <th style=\"padding-left: 1em; padding-right: 1em; text-align: center\">id</th>\n",
       "        <th style=\"padding-left: 1em; padding-right: 1em; text-align: center\">date</th>\n",
       "        <th style=\"padding-left: 1em; padding-right: 1em; text-align: center\">price</th>\n",
       "        <th style=\"padding-left: 1em; padding-right: 1em; text-align: center\">bedrooms</th>\n",
       "        <th style=\"padding-left: 1em; padding-right: 1em; text-align: center\">bathrooms</th>\n",
       "        <th style=\"padding-left: 1em; padding-right: 1em; text-align: center\">sqft_living</th>\n",
       "        <th style=\"padding-left: 1em; padding-right: 1em; text-align: center\">sqft_lot</th>\n",
       "        <th style=\"padding-left: 1em; padding-right: 1em; text-align: center\">floors</th>\n",
       "        <th style=\"padding-left: 1em; padding-right: 1em; text-align: center\">waterfront</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "        <td style=\"padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top\">7129300520</td>\n",
       "        <td style=\"padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top\">2014-10-13 00:00:00+00:00</td>\n",
       "        <td style=\"padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top\">221900.0</td>\n",
       "        <td style=\"padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top\">3.0</td>\n",
       "        <td style=\"padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top\">1.0</td>\n",
       "        <td style=\"padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top\">1180.0</td>\n",
       "        <td style=\"padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top\">5650</td>\n",
       "        <td style=\"padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top\">1</td>\n",
       "        <td style=\"padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top\">0</td>\n",
       "    </tr>\n",
       "</table>\n",
       "<table frame=\"box\" rules=\"cols\">\n",
       "    <tr>\n",
       "        <th style=\"padding-left: 1em; padding-right: 1em; text-align: center\">view</th>\n",
       "        <th style=\"padding-left: 1em; padding-right: 1em; text-align: center\">condition</th>\n",
       "        <th style=\"padding-left: 1em; padding-right: 1em; text-align: center\">grade</th>\n",
       "        <th style=\"padding-left: 1em; padding-right: 1em; text-align: center\">sqft_above</th>\n",
       "        <th style=\"padding-left: 1em; padding-right: 1em; text-align: center\">sqft_basement</th>\n",
       "        <th style=\"padding-left: 1em; padding-right: 1em; text-align: center\">yr_built</th>\n",
       "        <th style=\"padding-left: 1em; padding-right: 1em; text-align: center\">yr_renovated</th>\n",
       "        <th style=\"padding-left: 1em; padding-right: 1em; text-align: center\">zipcode</th>\n",
       "        <th style=\"padding-left: 1em; padding-right: 1em; text-align: center\">lat</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "        <td style=\"padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top\">0</td>\n",
       "        <td style=\"padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top\">3</td>\n",
       "        <td style=\"padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top\">7</td>\n",
       "        <td style=\"padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top\">1180</td>\n",
       "        <td style=\"padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top\">0</td>\n",
       "        <td style=\"padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top\">1955</td>\n",
       "        <td style=\"padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top\">0</td>\n",
       "        <td style=\"padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top\">98178</td>\n",
       "        <td style=\"padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top\">47.51123398</td>\n",
       "    </tr>\n",
       "</table>\n",
       "<table frame=\"box\" rules=\"cols\">\n",
       "    <tr>\n",
       "        <th style=\"padding-left: 1em; padding-right: 1em; text-align: center\">long</th>\n",
       "        <th style=\"padding-left: 1em; padding-right: 1em; text-align: center\">sqft_living15</th>\n",
       "        <th style=\"padding-left: 1em; padding-right: 1em; text-align: center\">sqft_lot15</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "        <td style=\"padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top\">-122.25677536</td>\n",
       "        <td style=\"padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top\">1340.0</td>\n",
       "        <td style=\"padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top\">5650.0</td>\n",
       "    </tr>\n",
       "</table>\n",
       "[1 rows x 21 columns]<br/>\n",
       "</div>"
      ],
      "text/plain": [
       "Columns:\n",
       "\tid\tstr\n",
       "\tdate\tdatetime\n",
       "\tprice\tfloat\n",
       "\tbedrooms\tfloat\n",
       "\tbathrooms\tfloat\n",
       "\tsqft_living\tfloat\n",
       "\tsqft_lot\tint\n",
       "\tfloors\tstr\n",
       "\twaterfront\tint\n",
       "\tview\tint\n",
       "\tcondition\tint\n",
       "\tgrade\tint\n",
       "\tsqft_above\tint\n",
       "\tsqft_basement\tint\n",
       "\tyr_built\tint\n",
       "\tyr_renovated\tint\n",
       "\tzipcode\tstr\n",
       "\tlat\tfloat\n",
       "\tlong\tfloat\n",
       "\tsqft_living15\tfloat\n",
       "\tsqft_lot15\tfloat\n",
       "\n",
       "Rows: 1\n",
       "\n",
       "Data:\n",
       "+------------+---------------------------+----------+----------+-----------+\n",
       "|     id     |            date           |  price   | bedrooms | bathrooms |\n",
       "+------------+---------------------------+----------+----------+-----------+\n",
       "| 7129300520 | 2014-10-13 00:00:00+00:00 | 221900.0 |   3.0    |    1.0    |\n",
       "+------------+---------------------------+----------+----------+-----------+\n",
       "+-------------+----------+--------+------------+------+-----------+-------+------------+\n",
       "| sqft_living | sqft_lot | floors | waterfront | view | condition | grade | sqft_above |\n",
       "+-------------+----------+--------+------------+------+-----------+-------+------------+\n",
       "|    1180.0   |   5650   |   1    |     0      |  0   |     3     |   7   |    1180    |\n",
       "+-------------+----------+--------+------------+------+-----------+-------+------------+\n",
       "+---------------+----------+--------------+---------+-------------+\n",
       "| sqft_basement | yr_built | yr_renovated | zipcode |     lat     |\n",
       "+---------------+----------+--------------+---------+-------------+\n",
       "|       0       |   1955   |      0       |  98178  | 47.51123398 |\n",
       "+---------------+----------+--------------+---------+-------------+\n",
       "+---------------+---------------+-----+\n",
       "|      long     | sqft_living15 | ... |\n",
       "+---------------+---------------+-----+\n",
       "| -122.25677536 |     1340.0    | ... |\n",
       "+---------------+---------------+-----+\n",
       "[1 rows x 21 columns]"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sales[0:1]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Splitting the data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "train_data, test_data = sales.random_split(.8,seed=0)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Simple linear regression algorithm - Part 1\n",
    "To calculate the intercept and the slope of the regression line"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# x is the input and y is the output\n",
    "def simple_linear_regression(x, y):\n",
    "    \n",
    "    # compute the sum of input and output\n",
    "    sum = x + y\n",
    "    \n",
    "    # compute the product of the output and the input and its sum\n",
    "    product = x * y\n",
    "    sum_of_product = product.sum()\n",
    "    \n",
    "    # compute the squared value of the input and its sum\n",
    "    x_squared = x * x\n",
    "    sum_x_squared = x_squared.sum()\n",
    "    \n",
    "    # use the formula for the slope\n",
    "    numerator = sum_of_product - ((x.sum() * y.sum()) / x.size())\n",
    "    denominator = sum_x_squared - ((x.sum() * x.sum()) / x.size())\n",
    "    slope = numerator / denominator\n",
    "    \n",
    "    # use the formula for the intercept\n",
    "    intercept = y.mean() - (slope * x.mean())\n",
    "    \n",
    "    return (intercept, slope)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We can test that our function works by passing it something where we know the answer. In particular we can generate a feature and then put the output exactly on a line: output = 1 + 1\\*input_feature then we know both our slope and intercept should be 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {
    "collapsed": false,
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0L, 1L, 2L, 3L, 4L]\n",
      "[1L, 2L, 3L, 4L, 5L]\n",
      "Intercept: 1.0\n",
      "Slope: 1\n"
     ]
    }
   ],
   "source": [
    "test_feature = graphlab.SArray(range(5))\n",
    "test_output = graphlab.SArray(1 + 1*test_feature)\n",
    "(test_intercept, test_slope) =  simple_linear_regression(test_feature, test_output)\n",
    "print test_feature\n",
    "print test_output\n",
    "print \"Intercept: \" + str(test_intercept)\n",
    "print \"Slope: \" + str(test_slope)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "So now it works let's build a regression model for predicting price based on sqft_living"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Intercept: -47116.0765749\n",
      "Slope: 281.958838568\n"
     ]
    }
   ],
   "source": [
    "sqft_intercept, sqft_slope = simple_linear_regression(train_data['sqft_living'], train_data['price'])\n",
    "\n",
    "print \"Intercept: \" + str(sqft_intercept)\n",
    "print \"Slope: \" + str(sqft_slope)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Simple linear regression algorithm - Part 2\n",
    "To calculate the predicted output"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "def get_regression_predictions(input_feature, intercept, slope):\n",
    "    # calculate the predicted values:\n",
    "    predicted_values = intercept + (slope * input_feature)\n",
    "    return predicted_values"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**What is the predicted price for a house with 2650 sqft?**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The estimated price for a house with 2650 squarefeet is $700074.85\n"
     ]
    }
   ],
   "source": [
    "my_house_sqft = 2650\n",
    "estimated_price = get_regression_predictions(my_house_sqft, sqft_intercept, sqft_slope)\n",
    "print \"The estimated price for a house with %d squarefeet is $%.2f\" % (my_house_sqft, estimated_price)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Residual Sum of Squares"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "RSS is the sum of the squares of the residuals which is the difference between the predicted output and the true output."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def get_residual_sum_of_squares(input_feature, actual_output, intercept, slope):\n",
    "    # First get the predictions\n",
    "    predicted_output = intercept + (slope * input_feature)\n",
    "\n",
    "    # then compute the residuals (since we are squaring it doesn't matter which order you subtract)\n",
    "    residuals = actual_output - predicted_output\n",
    "\n",
    "    # square the residuals and add them up\n",
    "    residuals_squared = residuals * residuals\n",
    "    residual_sum_squares = residuals_squared.sum()\n",
    "\n",
    "    return(residual_sum_squares)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The RSS of predicting Prices based on Square Feet is : 1.20191835632e+15\n"
     ]
    }
   ],
   "source": [
    "rss_prices_on_sqft = get_residual_sum_of_squares(train_data['sqft_living'], train_data['price'], sqft_intercept, sqft_slope)\n",
    "print 'The RSS of predicting Prices based on Square Feet is : ' + str(rss_prices_on_sqft)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Function to predict the squarefeet of a house from a given price"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def inverse_regression_predictions(output, intercept, slope):\n",
    "    # solve output = intercept + slope*input_feature for input_feature. Use this equation to compute the inverse predictions:\n",
    "    estimated_feature = (output - intercept) / slope\n",
    "    return estimated_feature"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**What is the estimated square-feet for a house costing $800,000?**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The estimated squarefeet for a house worth $800000.00 is 3004\n"
     ]
    }
   ],
   "source": [
    "my_house_price = 800000\n",
    "estimated_squarefeet = inverse_regression_predictions(my_house_price, sqft_intercept, sqft_slope)\n",
    "print \"The estimated squarefeet for a house worth $%.2f is %d\" % (my_house_price, estimated_squarefeet)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Estimate house price from no. of bedrooms"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Intercept: 109473.180469\n",
      "Slope: 127588.952175\n"
     ]
    }
   ],
   "source": [
    "# Estimate the slope and intercept for predicting 'price' based on 'bedrooms'\n",
    "bedrm_intercept, bedrm_slope = simple_linear_regression(train_data['bedrooms'], train_data['price'])\n",
    "print \"Intercept: \" + str(bedrm_intercept)\n",
    "print \"Slope: \" + str(bedrm_slope)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Test Linear Regression Algorithm for Square feet and Bedrooms Model"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Which model (square feet or bedrooms) has lowest RSS on TEST data?**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "275402936247141.3"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Compute RSS when using bedrooms on TEST data:\n",
    "get_residual_sum_of_squares(test_data['sqft_living'], test_data['price'], sqft_intercept, sqft_slope)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "493364582868287.94"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Compute RSS when using squarefeet on TEST data:\n",
    "get_residual_sum_of_squares(test_data['bedrooms'], test_data['price'], bedrm_intercept, bedrm_slope)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# So the Square feet model has a lower RSS than the Bedrooms model."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 2",
   "language": "python",
   "name": "python2"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 2
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}
