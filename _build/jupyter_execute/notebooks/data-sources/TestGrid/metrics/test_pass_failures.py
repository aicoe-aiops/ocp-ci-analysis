#!/usr/bin/env python
# coding: utf-8

# # Quantify test pass/failures
# 
# In this notebook, the key perfomance indicators that we would like to create greater visbility into and track over time is the percent of tests that passed/failed. This can help track and measure the effectiveness and success of our testing process.
# 
# By measuring this metric, we can observe the trend of test failures over time and a decrease in the percent of failures over time (or increase in percent of test passes), should correlate to improved test efficiencies, enhanced testing process and error free releases.  In this notebook, we derive the following metrics from the TestGrid dataset:
# 
# * total number of test cases
# * number of test cases passed
# * number of test cases failed
# * percent of tests that pass
# * percent of tests that fail
# 
# _Linked Issue: [issue 1](https://github.com/aicoe-aiops/ocp-ci-analysis/issues/114)_

# In[1]:


## Import libraries
import gzip
import json
import os
import pandas as pd
import datetime

from ipynb.fs.defs.metric_template import (
    testgrid_labelwise_encoding,
    CephCommunication,
    save_to_disk,
    read_from_disk,
)

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


# In[2]:


## Specify variables

METRIC_NAME = "test_pass_failures"

# Specify the path for input grid data
INPUT_DATA_PATH = "../../../../data/raw/testgrid_183.json.gz"

# Specify the path for output metric data
OUTPUT_DATA_PATH = f"../../../../data/processed/metrics/{METRIC_NAME}"

## CEPH Bucket variables
## Create a .env file on your local with the correct configs
s3_endpoint_url = os.getenv("S3_ENDPOINT")
s3_access_key = os.getenv("S3_ACCESS_KEY")
s3_secret_key = os.getenv("S3_SECRET_KEY")
s3_bucket = os.getenv("S3_BUCKET")
s3_path = os.getenv("S3_PROJECT_KEY", "ai4ci/testgrid/metrics")
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


# ## Metric Calculation
# We find all the tests which are failing i.e. have a status code of 12

# In[4]:


failures_list = testgrid_labelwise_encoding(testgrid_data, 12)


# In[5]:


len(failures_list)


# In[6]:


# Convert to dataframe
failures_df = pd.DataFrame(
    failures_list,
    columns=["timestamp", "tab", "grid", "test", "test_duration", "failure"],
)
failures_df = failures_df.drop(columns="test_duration")


# In[7]:


failures_df.head()


# We now find all the tests which are passing i.e. have a status code of 1

# In[8]:


passing_list = testgrid_labelwise_encoding(testgrid_data, 1)


# In[9]:


len(passing_list)


# In[10]:


# Convert to dataframe
passing_df = pd.DataFrame(
    passing_list,
    columns=["timestamp", "tab", "grid", "test", "test_duration", "passing"],
)
passing_df = passing_df.drop(columns="test_duration")


# In[11]:


passing_df.head()


# In[12]:


combined_df = pd.merge(
    failures_df,
    passing_df,
    on=["timestamp", "tab", "grid", "test"],
)
combined_df.head()


# In[13]:


## The following implements test pass/failure percentage metrics
## Moving forward, this will be aggregated in Superset
## For the sake of completeness, it is implmented here

no_tests = combined_df.test.count()
print("Total number of tests: %i" % (no_tests))
no_failures = combined_df.failure.sum()
print("Total number of failing tests: %i" % (no_failures))
test_failures_percentage = (
    (combined_df.failure.sum() / combined_df.test.count())
) * 100
print("Test failure percentage: %f" % (test_failures_percentage))
no_pass = combined_df.passing.sum()
print("Total number of passing tests: %i" % (no_pass))
test_pass_percentage = ((combined_df.passing.sum() / combined_df.passing.count())) * 100
print("Test pass percentage: %f" % (test_pass_percentage))


# ## Save results to Ceph or locally
# * Use the following helper function to save the data frame in a parquet format on the Ceph bucket if we are running in automation, and locally if not.

# In[14]:


timestamp = datetime.datetime.now()

if AUTOMATION:
    cc = CephCommunication(s3_endpoint_url, s3_access_key, s3_secret_key, s3_bucket)
    cc.upload_to_ceph(
        combined_df.head(1000000),
        s3_path,
        f"{METRIC_NAME}/{METRIC_NAME}-{timestamp.year}-{timestamp.month}-{timestamp.day}.parquet",
    )
else:
    save_to_disk(
        combined_df.head(1000000),
        OUTPUT_DATA_PATH,
        f"{METRIC_NAME}-{timestamp.year}-{timestamp.month}-{timestamp.day}.parquet",
    )


# In[15]:


## Sanity check to see if the dataset is the same
if AUTOMATION:
    sanity_check = cc.read_from_ceph(
        s3_path,
        f"{METRIC_NAME}/{METRIC_NAME}-{timestamp.year}-{timestamp.month}-{timestamp.day}.parquet",
    )
else:
    sanity_check = read_from_disk(
        OUTPUT_DATA_PATH,
        f"{METRIC_NAME}-{timestamp.year}-{timestamp.month}-{timestamp.day}.parquet",
    )

sanity_check


# ## Conclusion
# 
# This notebook computed the number of test passes, test failures and test pass/failure percentage metric. The dataframe saved on ceph can be used to generate aggregated views and visualizations.
