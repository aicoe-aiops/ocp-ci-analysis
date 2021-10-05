#!/usr/bin/env python
# coding: utf-8

# # Optimal Stopping Point for CI Tests
# 
# One of the machine learning explorations within the OpenShift CI Analysis project is predicting optimal stopping point for CI tests based on their test duration (runtimes) (see this [issue](https://github.com/aicoe-aiops/ocp-ci-analysis/issues/226) for more details). In a previous [notebook](../data-sources/TestGrid/metrics/time_to_fail.ipynb) we showed how to access the TestGrid data, and then performed initial data analysis as well as feature engineering on it. Furthermore, we also calculated the optimal stopping point by identifying the distribution of the `test_duration` values for different CI tests and comparing the distributions of passing/failing tests.
# 
# In this notebook, we will detect the optimal stopping point for different CI tests taken as inputs.

# In[1]:


## Import libraries
import os
import gzip
import json
import datetime
import itertools
import scipy  # noqa F401
from scipy.stats import (  # noqa F401
    invgauss,
    lognorm,
    pearson3,
    weibull_min,
    triang,
    beta,
    norm,
    weibull_max,
    uniform,
    gamma,
    expon,
)

from ipynb.fs.defs.osp_helper_functions import (
    CephCommunication,
    fit_distribution,
    standardize,
    filter_test_type,
    fetch_all_tests,
    best_distribution,
    optimal_stopping_point,
)
import warnings

warnings.filterwarnings("ignore")


# ## Ceph 
# Connection to Ceph for importing the TestGrid data

# In[2]:


## Specify variables
METRIC_NAME = "time_to_fail"

# Specify the path for input grid data
INPUT_DATA_PATH = "../../data/raw/testgrid_258.json.gz"

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


# ## Fetch all tests 
# 
# Using the function `fetch_all_tests`, we will fetch all passing and failing tests into two dataframes.

# In[4]:


# Fetch all failing tests i.e which have a status code of 12
failures_df = fetch_all_tests(testgrid_data, 12)


# In[5]:


failures_df.head()


# In[6]:


# Fetch all passing tests i.e which have a status code of 1
passing_df = fetch_all_tests(testgrid_data, 1)


# In[7]:


passing_df.head()


# ## Filter tests
# 
# After collecting the data for all passing and failing tests, we will move towards narrowing down to one test for which we would want to calculate the optimal stopping point. We will be using the test - `operator.Run multi-stage test e2e-aws-upgrade - e2e-aws-upgrade-openshift-e2e-test container test` and extract the data for this test.

# In[8]:


failures_test = filter_test_type(
    failures_df,
    "operator.Run multi-stage test e2e-aws-upgrade - e2e-aws-upgrade-openshift-e2e-test container test",
)
failures_test.head()


# In[9]:


passing_test = filter_test_type(
    passing_df,
    "operator.Run multi-stage test e2e-aws-upgrade - e2e-aws-upgrade-openshift-e2e-test container test",
)
passing_test.head()


# ## Fit Distribution
# 
# After extracting the data for one test, we would want to find the best distribution to perform optimal stopping point calculation. We find chi square and p-values to find the best distribution.

# In[10]:


failure_dist, failures_r = fit_distribution(failures_test, "test_duration", 0.99, 0.01)


# In[11]:


# Identify the best fit distribution from the failing test along with its corresponding distribution parameters
best_dist, parameters_failing = best_distribution(failure_dist, failures_r)


# In[12]:


# Identify the distributions for the passing test along with its corresponding distribution parameters
passing_dist, passing_r = fit_distribution(passing_test, "test_duration", 0.99, 0.01)


# In[13]:


passing_r.head()


# In[14]:


# Identify the best fit distribution from the passing test
best_distribution(passing_dist, passing_r)


# After finding the best distribution for failing distribution, we find the corresponding parameters for the same distribution in the passing distribution. 

# In[15]:


# Find the corresponding passing test distribution parameters for the
# best fit distribution identified from the failing test above
parameters_passing = passing_dist[passing_dist["Distribution Names"] == best_dist][
    "Parameters"
].values
parameters_passing = list(itertools.chain(*parameters_passing))


# In[16]:


# Standardize the features by removing the mean and scaling to unit variance
y_std_failing, len_y_failing, y_failing = standardize(
    failures_test, "test_duration", 0.99, 0.01
)


# In[17]:


# Standardize the features by removing the mean and scaling to unit variance
y_std_passing, len_y_passing, y_passing = standardize(
    passing_test, "test_duration", 0.99, 0.01
)


# ## Optimal Stopping Point Calculation
# 
# Let's move forward to find the optimal stopping point for the test by passing the best distribution name, failing and passing distributions and the corresponding distribution parameters.

# In[18]:


osp = optimal_stopping_point(
    best_dist,
    y_std_failing,
    y_failing,
    parameters_failing,
    y_std_passing,
    y_passing,
    parameters_passing,
)


# In[19]:


# Optimat Stopping Point for `operator.Run multi-stage test e2e-aws-upgrade
# - e2e-aws-upgrade-openshift-e2e-test container test`
osp


# This tells us that the optimal stopping point should be at test duration run length of 104.39 seconds.

# ## Conclusion
# In this notebook we were able to:
# 
# * Fetch the data for all passing and failing tests
# * Filter the data for the test - `operator.Run multi-stage test e2e-aws-upgrade - e2e-aws-upgrade-openshift-e2e-test container test`
# * Find the best distribution for the test
# * Find the optimal stopping point for the test
