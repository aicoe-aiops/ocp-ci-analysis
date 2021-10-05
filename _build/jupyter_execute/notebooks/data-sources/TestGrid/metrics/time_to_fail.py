#!/usr/bin/env python
# coding: utf-8

# # Time to Fail
# 
# One of the machine learning requests that we would like to create greater visibility into and for a given test, to build a model that can predict an optimal stopping point, beyond which a test is more likely to result in a failure.
# 
# In order to achieve the above, we would be looking into the data for all the passing and failed tests and find the distribution type for the `test_duration` metric. The `test_duration` metric tracks the time it took for a test to complete its execution. We can visualize the distribution of the test_duration metric across various testgrid dashboards and jobs. Based on the distribution type identified, we can find a point after which the test has a higher probability of failing.
# 
# Linked issue(s) : [Issue1](https://github.com/aicoe-aiops/ocp-ci-analysis/issues/333), [Issue2](https://github.com/aicoe-aiops/ocp-ci-analysis/issues/226)

# In[1]:


import json
import gzip
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import scipy.stats
import itertools
from intersect import intersection
from sklearn.preprocessing import StandardScaler
from scipy.stats import invgauss, lognorm, pearson3, weibull_min, triang, beta

from ipynb.fs.defs.metric_template import testgrid_labelwise_encoding
from ipynb.fs.defs.metric_template import CephCommunication
from dotenv import load_dotenv, find_dotenv
import warnings

warnings.filterwarnings("ignore")

load_dotenv(find_dotenv())


# ## Ceph
# Connection to Ceph for importing the TestGrid data

# In[2]:


## Specify variables

METRIC_NAME = "time_to_fail"

# Specify the path for input grid data
INPUT_DATA_PATH = "../../../../data/raw/testgrid_258.json.gz"

# Specify the path for output metric data
OUTPUT_DATA_PATH = f"../../../../data/processed/metrics/{METRIC_NAME}"

## CEPH Bucket variables
## Create a .env file on your local with the correct configs
s3_endpoint_url = os.getenv("S3_ENDPOINT")
s3_access_key = os.getenv("S3_ACCESS_KEY")
s3_secret_key = os.getenv("S3_SECRET_KEY")
s3_bucket = os.getenv("S3_BUCKET")
s3_path = os.getenv("S3_PROJECT_KEY", "metrics")
s3_input_data_path = "raw_data"

# Specify whether or not we are running this as a notebook or part of an automation pipeline.
AUTOMATION = os.getenv("IN_AUTOMATION")


# In[3]:


## Import data
timestamp = datetime.datetime.today()

if AUTOMATION:
    filename = f"testgrid_{timestamp.day}{timestamp.month}.json"
    cc = CephCommunication(s3_endpoint_url, s3_access_key, s3_secret_key, s3_bucket)
    s3_object = cc.s3_resource.Object(s3_bucket, f"{s3_input_data_path}/{filename}")
    file_content = s3_object.get()["Body"].read().decode("utf-8")
    testgrid_data = json.loads(file_content)

else:
    with gzip.open(INPUT_DATA_PATH, "rb") as read_file:
        testgrid_data = json.load(read_file)


# ## Helper Functions

# In[4]:


# Function to filter the df for a specified test
def filter_test_type(df, test):
    failures_test = df[df["test"] == test]
    failures_test = failures_test.reset_index(drop=True)
    return failures_test


# In[5]:


def standardize(df, column, pct, pct_lower):
    """
    Function to standardize the features by removing the mean
    and scaling to unit variance using StandardScaler library.

    Returns standandardized feature, length of the feature
    and the original feature.
    """
    sc = StandardScaler()
    y = df[column][df[column].notnull()].to_list()
    y.sort()
    len_y = len(y)
    y = y[int(pct_lower * len_y) : int(len_y * pct)]
    len_y = len(y)
    yy = [[x] for x in y]
    sc.fit(yy)
    y_std = sc.transform(yy)
    y_std = y_std.flatten()
    return y_std, len_y, y


# In[6]:


def fit_distribution(df, column, pct, pct_lower):
    """
    This function helps to list out the chi-square statistics for each
    distribution and further sorts them to find the best distribution.

    Returns a table that contains sorted chi-square values as well as
    the parameters such as mu (shape), loc (location) and scale for each
    distribution.
    """
    # Set up list of candidate distributions to use
    y_std, size, y_org = standardize(df, column, pct, pct_lower)
    dist_names = [
        "weibull_min",
        "norm",
        "weibull_max",
        "beta",
        "invgauss",
        "uniform",
        "gamma",
        "expon",
        "lognorm",
        "pearson3",
        "triang",
    ]

    chi_square_statistics = []

    # 50 bins
    percentile_bins = np.linspace(0, 100, 50)
    percentile_cutoffs = np.percentile(y_std, percentile_bins)
    observed_frequency, bins = np.histogram(y_std, bins=percentile_cutoffs)
    cum_observed_frequency = np.cumsum(observed_frequency)
    # Data frame to store results
    dist_param = pd.DataFrame()
    dist_param["Distribution Names"] = dist_names
    param_list = []

    # Loop through candidate distributions
    for distribution in dist_names:
        # Set up distribution and get fitted distribution parameters
        dist = getattr(scipy.stats, distribution)
        param = dist.fit(y_std)
        param_list.append(param)

        # Get expected counts in percentile bins
        # cdf of fitted distribution across bins
        cdf_fitted = dist.cdf(percentile_cutoffs, *param)
        expected_frequency = []
        for bin in range(len(percentile_bins) - 1):
            expected_cdf_area = cdf_fitted[bin + 1] - cdf_fitted[bin]
            expected_frequency.append(expected_cdf_area)

        # Chi-square Statistics
        expected_frequency = np.array(expected_frequency) * size
        cum_expected_frequency = np.cumsum(expected_frequency)
        ss = scipy.stats.chisquare(
            f_obs=cum_observed_frequency, f_exp=cum_expected_frequency
        )
        chi_square_statistics.append(ss)

    # Append results to data frame
    dist_param["Parameters"] = param_list
    dist_param.set_index("Distribution Names")
    # Sort by minimum ch-square statistics
    results = pd.DataFrame()
    results["Distribution"] = dist_names
    results["chi_square and p-value"] = chi_square_statistics
    results.sort_values(["chi_square and p-value"], inplace=True)

    print("\nDistributions listed by Betterment of fit:")
    print("............................................")
    print(results)
    return dist_param, results


# ## Calculation
# Let's fetch all the tests which are "Passing" and "Failing".

# ### Failing Tests

# In[7]:


# We will now fetch all the tests which are failing i.e. have a status code of 12.
failures_list = testgrid_labelwise_encoding(testgrid_data, 12, overall_only=False)


# In[8]:


# Convert to dataframe
failures_df = pd.DataFrame(
    failures_list,
    columns=["timestamp", "tab", "grid", "test", "test_duration", "failure"],
)
failures_df.head()


# In[9]:


len(failures_df)


# In[10]:


# We will drop all the rows having NaN values
failures_df = failures_df.dropna()
len(failures_df)


# In[11]:


# We will now filter the df for extracting only the tests that are failing
failures_df = failures_df[failures_df["failure"]]
len(failures_df)


# In[12]:


failures_df.head()


# ### Passing Tests

# In[13]:


# We will now fetch all the tests which are passing i.e. have a status code of 1.
passing_list = testgrid_labelwise_encoding(testgrid_data, 1, overall_only=False)


# In[14]:


# Convert to dataframe
passing_df = pd.DataFrame(
    passing_list,
    columns=["timestamp", "tab", "grid", "test", "test_duration", "passing"],
)
passing_df.head()


# In[15]:


len(passing_df)


# In[16]:


# We will drop all the rows having NaN values
passing_df = passing_df.dropna()
len(passing_df)


# In[17]:


# We will now filter the df for extracting only the tests that are passing
passing_df = passing_df[passing_df["passing"]]
len(passing_df)


# In[18]:


passing_df.head()


# ## Probability Distribution of Data
# Data Distribution is a function that lists out all possible values the data can take. It can be a continuous or discrete data distribution. Several known standard Probability Distribution functions provide probabilities of occurrence of different possible outcomes in an experiment. Some well-known probability distributions are Normal, Log-Normal, Beta, Gamma, etc. which have a standard form.
# 
# We will try to approximate the distributions of the `test_duration` variable and also check its Goodness of fit for different TestGrid tests across all TestGrid dashboards and grids. Based on the type of distribution identified, we can calculate the probability of the test failing.

# Let's see what are the common failing and passing test types and identify both the passing and failing distribution for the top 2 test types.

# In[19]:


failures_df["test"].value_counts()


# In[20]:


passing_df["test"].value_counts()


# Now let's find the common test types which are both failing and passing

# In[21]:


combined = pd.merge(
    failures_df,
    passing_df,
    on=["tab", "grid", "test"],
)


# In[22]:


combined["test"].value_counts()


# ### Failure Distribution
# 
# Let's identify the distribution type for the following 2 tests:
# * openshift-tests.[sig-arch][Feature:ClusterUpgrade] Cluster should remain functional during upgrade [Disruptive] [Serial]
# * operator.Run multi-stage test e2e-aws-upgrade - e2e-aws-upgrade-openshift-e2e-test container test
# 
# #### Test: "openshift-tests.[sig-arch][Feature:ClusterUpgrade] Cluster should remain functional during upgrade [Disruptive] [Serial]"

# In[23]:


# Filter df for the "openshift-tests.[sig-arch][Feature:ClusterUpgrade] Cluster should remain functional
# during upgrade [Disruptive] [Serial]" test
failures_test1 = filter_test_type(
    failures_df,
    "openshift-tests.[sig-arch][Feature:ClusterUpgrade] Cluster should remain functional "
    "during upgrade [Disruptive] [Serial]",
)
failures_test1.head()


# In[24]:


# Let's plot a histogram to visualize the distribution of the observed `test_duration` data points
failures_test1["test_duration"].hist()


# In[25]:


# Identify the distribution
d1_failing, r1_failing = fit_distribution(failures_test1, "test_duration", 0.99, 0.01)


# In[26]:


# Print the parameters for the distributions which are the mu (shape), loc (location)
# and scale parameters
print(d1_failing)


# We see that the top 2 distributions based on betterment of fit are **Inverse Gaussian** distribution and **Log Normal** distribution. Let's plot the graphs for these two distributions.

# In[27]:


# Fetch the parameters required for respective distribution types to visualize the density plots
invgauss_param_failing1 = list(
    d1_failing[d1_failing["Distribution Names"] == "invgauss"]["Parameters"].values
)
# Flatten list
invgauss_param_failing1 = list(itertools.chain(*invgauss_param_failing1))
lognorm_param_failing1 = list(
    d1_failing[d1_failing["Distribution Names"] == "lognorm"]["Parameters"].values
)
# Flatten list
lognorm_param_failing1 = list(itertools.chain(*lognorm_param_failing1))


# In[28]:


y_std_failing1, len_y_failing1, y_failing1 = standardize(
    failures_test1, "test_duration", 0.99, 0.01
)


# In[29]:


# Plot the distributions
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9, 5))
axes[0].hist(y_failing1)
axes[0].set_xlabel("Test Duration\n\nHistogram plot of Oberseved Data")
axes[0].set_ylabel("Frequency")
# Plot the density plot for Inverse Gaussian distribution by passing the mu (shape parameter), loc (location)
# and scale parameters obtained from above
axes[1].plot(
    y_failing1,
    invgauss.pdf(
        y_std_failing1,
        invgauss_param_failing1[0],
        invgauss_param_failing1[1],
        invgauss_param_failing1[2],
    ),
)
axes[1].set_xlabel("Test Duration\n\nInverse Gaussian Distribution")
axes[1].set_ylabel("pdf")
# Plot the density plot for Log Normal distribution by passing the mu (shape), loc (location) and
# scale parameters obtained from above
axes[2].plot(
    y_failing1,
    lognorm.pdf(
        y_std_failing1,
        lognorm_param_failing1[0],
        lognorm_param_failing1[1],
        lognorm_param_failing1[2],
    ),
)
axes[2].set_xlabel("Test Duration\n\nLog Normal Distribution")
axes[2].set_ylabel("pdf")
fig.tight_layout()


# The Histogram plot shows the distribution of test duration values over all the data points. The Inverse Gaussian and Log Normal graphs are density plots which are a smoothed, continuous version of a histogram estimated from the data. It plots the probability density function (along Y-axis) vs the test duration values (along X-axis). Probability density plots are used to understand data distribution for a continuous variable and we want to know the likelihood (or probability) of obtaining a range of values that the continuous variable can assume. The area under the curve contains the probabilities for the test duration values.

# #### Test: "operator.Run multi-stage test e2e-aws-upgrade - e2e-aws-upgrade-openshift-e2e-test container test"

# In[30]:


# Filter df for the "operator.Run multi-stage test e2e-aws-upgrade - e2e-aws-upgrade-openshift-e2e-test
# container test" test
failures_test2 = filter_test_type(
    failures_df,
    "operator.Run multi-stage test e2e-aws-upgrade - "
    "e2e-aws-upgrade-openshift-e2e-test container test",
)
failures_test2.head()


# In[31]:


# Let's plot a histogram to visualize the distribution of the observed `test_duration` data points
failures_test2["test_duration"].hist()


# In[32]:


# Identify the distribution
d2_failing, r2_failing = fit_distribution(failures_test2, "test_duration", 0.99, 0.01)


# In[33]:


# Print the parameters for the distributions which are the mu (shape), loc (location)
# and scale parameters
print(d2_failing)


# We see that the top 2 distributions based on betterment of fit are **Pearson** distribution and **Beta** distribution. Let's plot the graphs for these two distributions.

# In[34]:


# Fetch the parameters required for respective distribution types to visualize the density plots
pearson_param_failing2 = d2_failing[d2_failing["Distribution Names"] == "pearson3"][
    "Parameters"
].values
# Flatten list
pearson_param_failing2 = list(itertools.chain(*pearson_param_failing2))
beta_param_failing2 = d2_failing[d2_failing["Distribution Names"] == "beta"][
    "Parameters"
].values
# Flatten list
beta_param_failing2 = list(itertools.chain(*beta_param_failing2))


# In[35]:


y_std_failing2, len_y_failing2, y_failing2 = standardize(
    failures_test2, "test_duration", 0.99, 0.01
)


# In[36]:


# Plot the distributions
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9, 5))
axes[0].hist(y_failing2)
axes[0].set_xlabel("Test Duration\n\nHistogram plot of Oberseved Data")
axes[0].set_ylabel("Frequency")
# Plot the density plot for Pearson distribution by passing the mu (shape parameter), loc (location)
# and scale parameters obtained from above
axes[1].plot(
    y_failing2,
    pearson3.pdf(
        y_std_failing2,
        pearson_param_failing2[0],
        pearson_param_failing2[1],
        pearson_param_failing2[2],
    ),
)
axes[1].set_xlabel("Test Duration\n\nPearson Distribution")
axes[1].set_ylabel("pdf")
# Plot the density plot for Beta distribution by passing the mu (shape), loc (location) and
# scale parameters obtained from above
axes[2].plot(
    y_failing2,
    beta.pdf(
        y_std_failing2,
        beta_param_failing2[0],
        beta_param_failing2[1],
        beta_param_failing2[2],
    ),
)
axes[2].set_xlabel("Test Duration\n\nBeta Distribution")
axes[2].set_ylabel("pdf")
fig.tight_layout()


# The Histogram plot shows the distribution of test duration values over all the data points. The Pearson and Beta graphs are density plots which are a smoothed, continuous version of a histogram estimated from the data. It plots the probability density function (along Y-axis) vs the test duration values (along X-axis). Probability density plots are used to understand data distribution for a continuous variable and we want to know the likelihood (or probability) of obtaining a range of values that the continuous variable can assume. The area under the curve contains the probabilities for the test duration values.

# ### Passing Distribution
# 
# Let's identify the passing distribution type for the following 2 tests:
# * openshift-tests.[sig-arch][Feature:ClusterUpgrade] Cluster should remain functional during upgrade [Disruptive] [Serial]
# * operator.Run multi-stage test e2e-aws-upgrade - e2e-aws-upgrade-openshift-e2e-test container test
# 
# #### Test: "openshift-tests.[sig-arch][Feature:ClusterUpgrade] Cluster should remain functional during upgrade [Disruptive] [Serial]"

# In[37]:


# Filter df for the "openshift-tests.[sig-arch][Feature:ClusterUpgrade] Cluster should remain
# functional during upgrade [Disruptive] [Serial]" test
passing_test1 = filter_test_type(
    passing_df,
    "openshift-tests.[sig-arch][Feature:ClusterUpgrade] Cluster should remain "
    "functional during upgrade [Disruptive] [Serial]",
)
passing_test1.head()


# In[38]:


# Let's plot a histogram to visualize the distribution of the observed `test_duration` data points
passing_test1["test_duration"].hist()


# In[39]:


# Identify the distribution
d1_passing, r1_passing = fit_distribution(passing_test1, "test_duration", 0.99, 0.01)


# In[40]:


# Print the parameters for the distributions which are the mu (shape), loc (location)
# and scale parameters
print(d1_passing)


# We see that the top 2 distributions based on betterment of fit are **Pearson** distribution and **Weibull Min** distribution. Let's plot the graphs for these two distributions.

# In[41]:


# Fetch the parameters required for respective distribution types to visualize the density plots
pearson_param_passing1 = d1_passing[d1_passing["Distribution Names"] == "pearson3"][
    "Parameters"
].values
# Flatten list
pearson_param_passing1 = list(itertools.chain(*pearson_param_passing1))
weibull_param_passing1 = d1_passing[d1_passing["Distribution Names"] == "weibull_min"][
    "Parameters"
].values
# Flatten list
weibull_param_passing1 = list(itertools.chain(*weibull_param_passing1))


# In[42]:


y_std_passing1, len_y_passing1, y_passing1 = standardize(
    passing_test1, "test_duration", 0.99, 0.01
)


# In[43]:


# Plot the distributions
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9, 5))
axes[0].hist(y_passing1)
axes[0].set_xlabel("Test Duration\n\nHistogram plot of Oberseved Data")
axes[0].set_ylabel("Frequency")
# Plot the density plot for Pearson distribution by passing the mu (shape parameter), loc (location)
# and scale parameters obtained from above
axes[1].plot(
    y_passing1,
    pearson3.pdf(
        y_std_passing1,
        pearson_param_passing1[0],
        pearson_param_passing1[1],
        pearson_param_passing1[2],
    ),
)
axes[1].set_xlabel("Test Duration\n\nPearson Distribution")
axes[1].set_ylabel("pdf")
# Plot the density plot for Weibull Min distribution by passing the mu (shape), loc (location) and
# scale parameters obtained from above
axes[2].plot(
    y_passing1,
    weibull_min.pdf(
        y_std_passing1,
        weibull_param_passing1[0],
        weibull_param_passing1[1],
        weibull_param_passing1[2],
    ),
)
axes[2].set_xlabel("Test Duration\n\nWeibull Min Distribution")
axes[2].set_ylabel("pdf")
fig.tight_layout()


# The Histogram plot shows the distribution of test duration values over all the data points. The Pearson and Weibull Min graphs are density plots which are a smoothed, continuous version of a histogram estimated from the data. It plots the probability density function (along Y-axis) vs the test duration values (along X-axis). Probability density plots are used to understand data distribution for a continuous variable and we want to know the likelihood (or probability) of obtaining a range of values that the continuous variable can assume. The area under the curve contains the probabilities for the test duration values.

# #### Test: "operator.Run multi-stage test e2e-aws-upgrade - e2e-aws-upgrade-openshift-e2e-test container test"

# In[44]:


# Filter df for the "operator.Run multi-stage test e2e-aws-upgrade - e2e-aws-upgrade-openshift-e2e-test
# container test" test
passing_test2 = filter_test_type(
    passing_df,
    "operator.Run multi-stage test e2e-aws-upgrade - "
    "e2e-aws-upgrade-openshift-e2e-test container test",
)
passing_test2.head()


# In[45]:


# Let's plot a histogram to visualize the distribution of the observed `test_duration` data points
passing_test2["test_duration"].hist()


# In[46]:


# Identify the distribution
d2_passing, r2_passing = fit_distribution(passing_test2, "test_duration", 0.99, 0.01)


# In[47]:


# Print the parameters for the distributions which are the mu (shape), loc (location)
# and scale parameters
print(d2_passing)


# We see that the top 2 distributions based on betterment of fit are **Pearson** distribution and **Triangular** distribution. Let's plot the graphs for these two distributions.

# In[48]:


# Fetch the parameters required for respective distribution types to visualize the density plots
pearson_param_passing2 = d2_passing[d2_passing["Distribution Names"] == "pearson3"][
    "Parameters"
].values
# Flatten list
pearson_param_passing2 = list(itertools.chain(*pearson_param_passing2))
triang_param_passing2 = d2_passing[d2_passing["Distribution Names"] == "triang"][
    "Parameters"
].values
# Flatten list
triang_param_passing2 = list(itertools.chain(*triang_param_passing2))


# In[49]:


y_std_passing2, len_y_passing2, y_passing2 = standardize(
    passing_test2, "test_duration", 0.99, 0.01
)


# In[50]:


# Plot the distributions
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9, 5))
axes[0].hist(y_passing2)
axes[0].set_xlabel("Test Duration\n\nHistogram plot of Oberseved Data")
axes[0].set_ylabel("Frequency")
# Plot the density plot for Pearson distribution by passing the mu (shape parameter), loc (location)
# and scale parameters obtained from above
axes[1].plot(
    y_passing2,
    pearson3.pdf(
        y_std_passing2,
        pearson_param_passing2[0],
        pearson_param_passing2[1],
        pearson_param_passing2[2],
    ),
)
axes[1].set_xlabel("Test Duration\n\nPearson Distribution")
axes[1].set_ylabel("pdf")
# Plot the density plot for Triangular distribution by passing the mu (shape), loc (location) and
# scale parameters obtained from above
axes[2].plot(
    y_passing2,
    triang.pdf(
        y_std_passing2,
        triang_param_passing2[0],
        triang_param_passing2[1],
        triang_param_passing2[2],
    ),
)
axes[2].set_xlabel("Test Duration\n\nTriangular Distribution")
axes[2].set_ylabel("pdf")
fig.tight_layout()


# The Histogram plot shows the distribution of test duration values over all the data points. The Pearson and Triangular graphs are density plots which are a smoothed, continuous version of a histogram estimated from the data. It plots the probability density function (along Y-axis) vs the test duration values (along X-axis). Probability density plots are used to understand data distribution for a continuous variable and we want to know the likelihood (or probability) of obtaining a range of values that the continuous variable can assume. The area under the curve contains the probabilities for the test duration values.

# ## Optimal Stopping Point
# Next, we would want to look into these distribution types to determine a point after which the test has a higher probability of failing.
# 
# ### Test: "openshift-tests.[sig-arch][Feature:ClusterUpgrade] Cluster should remain functional during upgrade [Disruptive] [Serial]"
# *Distribution Type*: **Log Normal**

# For the test "openshift-tests.[sig-arch][Feature:ClusterUpgrade] Cluster should remain functional during upgrade [Disruptive] [Serial]", let's look at the `Log Normal` distribution type.
# 
# To find the optimal stopping point, we will find the intersection points for the passing and failing distribution curves using the `intersect` library. We will then consider the largest `x` co-ordinate value (i.e the test duration value) as the corresponding optimal stopping point.

# In[51]:


# Fetch the parameters for the log normal distribution of passing test1
lognorm_param_passing1 = d1_passing[d1_passing["Distribution Names"] == "lognorm"][
    "Parameters"
].values
# Flatten list
lognorm_param_passing1 = list(itertools.chain(*lognorm_param_passing1))


# In[52]:


# Obtain the intersection points between the distribution curves
x1, y1 = intersection(
    y_failing1,
    lognorm.pdf(
        y_std_failing1,
        lognorm_param_failing1[0],
        lognorm_param_failing1[1],
        lognorm_param_failing1[2],
    ),
    y_passing1,
    lognorm.pdf(
        y_std_passing1,
        lognorm_param_passing1[0],
        lognorm_param_passing1[1],
        lognorm_param_passing1[2],
    ),
)
# Print the x co-ordinates of the intersection points which corresponds to the test duration values
print(x1)


# In[53]:


fig, ax = plt.subplots()
ax.plot(
    y_failing1,
    lognorm.pdf(
        y_std_failing1,
        lognorm_param_failing1[0],
        lognorm_param_failing1[1],
        lognorm_param_failing1[2],
    ),
    label="Failure Distribution",
)
ax.plot(
    y_passing1,
    lognorm.pdf(
        y_std_passing1,
        lognorm_param_passing1[0],
        lognorm_param_passing1[1],
        lognorm_param_passing1[2],
    ),
    label="Passing Distribution",
)
ax.set_xlabel("Test Duration")
ax.set_ylabel("pdf")
ax.set_title("Test Duration vs Probability Density Function")
# vertical intersection point corresponding to largest x co-ordinate value
ax.axvline(x=max(x1), color="r", label="Optimal Stopping Point")
fig.tight_layout()
plt.legend()
plt.show()


# From the above graph, we see that passing tests (orange line) have a narrower range and smaller std, with an expected run time of between 20 and 110 seconds. Failing tests (blue line) are much less reliable with a run time spanning from 0 to 240 seconds. From this graph, we can claim that a test run longer than ~72 seconds starts to have an increased probability of being a failure and hence can be considered as an optimal stopping point.

# ### Test: "operator.Run multi-stage test e2e-aws-upgrade - e2e-aws-upgrade-openshift-e2e-test container test"
# *Distribution Type*: **Person**

# For the test "operator.Run multi-stage test e2e-aws-upgrade - e2e-aws-upgrade-openshift-e2e-test container test", let's look at the `Pearson` distribution type.
# 
# To find the optimal stopping point, we will find the intersection points for the passing and failing distribution curves using the `intersect` library. We will then consider the largest `x` co-ordinate value (i.e the test duration value) as the corresponding optimal stopping point.

# In[54]:


# Obtain the intersection points between the distribution curves
x2, y2 = intersection(
    y_failing2,
    pearson3.pdf(
        y_std_failing2,
        pearson_param_failing2[0],
        pearson_param_failing2[1],
        pearson_param_failing2[2],
    ),
    y_passing2,
    pearson3.pdf(
        y_std_passing2,
        pearson_param_passing2[0],
        pearson_param_passing2[1],
        pearson_param_passing2[2],
    ),
)
# Print the x co-ordinates of the intersection points which corresponds to the test duration values
print(x2)


# In[55]:


fig, ax = plt.subplots()
ax.plot(
    y_failing2,
    pearson3.pdf(
        y_std_failing2,
        pearson_param_failing2[0],
        pearson_param_failing2[1],
        pearson_param_failing2[2],
    ),
    label="Failure Distribution",
)
ax.plot(
    y_passing2,
    pearson3.pdf(
        y_std_passing2,
        pearson_param_passing2[0],
        pearson_param_passing2[1],
        pearson_param_passing2[2],
    ),
    label="Passing Distribution",
)
ax.set_xlabel("Test Duration")
ax.set_ylabel("pdf")
ax.set_title("Test Duration vs Probability Density Function")
# vertical intersection point corresponding to largest x co-ordinate value
ax.axvline(x=max(x2), color="r", label="Optimal Stopping Point")
fig.tight_layout()
plt.legend()
plt.show()


# From the above graph, we see that passing tests (orange line) have a narrower range and smaller std, with an expected run time of between 40 and 120 seconds. Failing tests (blue line) are much less reliable with a run time spanning from 0 to 160 seconds. From this graph, we can claim that a test run longer than ~104 seconds starts to have an increased probability of being a failure and hence can be considered as an optimal stopping point.

# ## Conclusion:
# 
# In this notebook, we have filtered the TestGrid data for failing and passing tests and identified the distributions for different TestGrid tests for the feature `test_duration`. We have observed that different TestGrid tests have different types of distributions. Based on the distribution type, we performed Chi-Square statistics for each distribution and further sorted them to find the best distribution. We then determine an optimal stopping point by plotting the intersection points between the passing and failing test distributions. For future work, we aim to develop an ML model to predict an optimal stopping point for each test considering the distribution type so as to make appropriate predictions.
