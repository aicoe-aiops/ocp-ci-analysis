#!/usr/bin/env python
# coding: utf-8

# # Tests Blocked and Timed Out Tests
# 
# This notebook is an extension to the [number_of_flakes](number_of_flakes.ipynb) notebook. In this notebook, the key perfomance indicators that we would like to create greater visbility into and track over time is the percent of tests that got blocked or were timed out. By observing the above metrics and tracking them wrt time, we will be able to quantify the efficiency of our testing platforms.
# 
# * number and percent of tests blocked
# * number and percent of tests timed out
# 
# Linked issue : [issue](https://github.com/aicoe-aiops/ocp-ci-analysis/issues/114)

# In[1]:


import gzip
import json
import os
import pandas as pd
import datetime

from ipynb.fs.defs.metric_template import testgrid_labelwise_encoding
from ipynb.fs.defs.metric_template import CephCommunication
from ipynb.fs.defs.metric_template import save_to_disk, read_from_disk
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


# In[2]:


## Specify variables

METRIC_NAME = "blocked_timed_out"

# Specify the path for input grid data,
INPUT_DATA_PATH = "../../../../data/raw/testgrid_183.json.gz"

# Specify the path for output metric data
OUTPUT_DATA_PATH = f"../../../../data/processed/metrics/{METRIC_NAME}"

## CEPH Bucket variables
## Create a .env file on your local with the correct configs,
s3_endpoint_url = os.getenv("S3_ENDPOINT")
s3_access_key = os.getenv("S3_ACCESS_KEY")
s3_secret_key = os.getenv("S3_SECRET_KEY")
s3_bucket = os.getenv("S3_BUCKET")
s3_path = os.getenv("S3_PROJECT_KEY", "ai4ci/testgrid/metrics")
s3_input_data_path = "raw_data"
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
# 1. **Blocked Tests** : Finding all the tests that are blocked i.e. tests with the status code 8.

# In[4]:


blocked_tests_list = testgrid_labelwise_encoding(testgrid_data, 8)[0:1000000]


# In[5]:


len(blocked_tests_list)


# In[6]:


blocked_tests_list[0]


# In[7]:


# Convert to dataframe
blocked_tests_df = pd.DataFrame(
    blocked_tests_list,
    columns=["timestamp", "tab", "grid", "test", "test_duration", "test_blocked"],
)
blocked_tests_df.head()


# * **Timed Out Tests** : Finding all the tests that are timed out i.e. tests with the status code 9.

# In[8]:


timed_out_tests_list = testgrid_labelwise_encoding(testgrid_data, 9)[0:1000000]


# In[9]:


len(timed_out_tests_list)


# In[10]:


timed_out_tests_list[0]


# In[11]:


# Convert to dataframe
timed_out_tests_df = pd.DataFrame(
    timed_out_tests_list,
    columns=["timestamp", "tab", "grid", "test", "test_duration", "test_timed_out"],
)
timed_out_tests_df.head()


# In[12]:


no_tests = blocked_tests_df.test.count()
print("Total number of tests: %i" % (no_tests))
no_blocked = blocked_tests_df.test_blocked.sum()
print("Total number of tests blocked: %i" % (no_blocked))
test_blocked_percentage = (
    (blocked_tests_df.test_blocked.sum() / blocked_tests_df.test.count())
) * 100
print("Tests blocked percentage: %f" % (test_blocked_percentage))
no_timed_out = timed_out_tests_df.test_timed_out.sum()
print("Total number of timed out tests: %i" % (no_timed_out))
test_timed_out_percentage = (
    (timed_out_tests_df.test_timed_out.sum() / timed_out_tests_df.test.count())
) * 100
print("Test timed out percentage: %f" % (test_timed_out_percentage))


# In[13]:


combined = pd.merge(
    blocked_tests_df,
    timed_out_tests_df,
    on=["timestamp", "tab", "grid", "test", "test_duration"],
)


# In[14]:


combined.drop(columns="test_duration", inplace=True)
combined.head()


# ## Save to Ceph or Local

# In[15]:


if AUTOMATION:
    cc = CephCommunication(s3_endpoint_url, s3_access_key, s3_secret_key, s3_bucket)
    cc.upload_to_ceph(
        combined,
        s3_path,
        f"{METRIC_NAME}/{METRIC_NAME}-{timestamp.year}-{timestamp.month}-{timestamp.day}.parquet",
    )
else:
    save_to_disk(
        combined,
        OUTPUT_DATA_PATH,
        f"{METRIC_NAME}-{timestamp.year}-{timestamp.month}-{timestamp.day}.parquet",
    )


# In[16]:


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


# ## Conclusion :
# 
# This notebook computed number of timed out tests and blocked tests. The combined dataframe is then saved on ceph and can be used to generate views and visualizations.
