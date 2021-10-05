#!/usr/bin/env python
# coding: utf-8

# # Correlated test failure sets per test and average size of correlation sets
# 
# This notebook outputs 2 artifacts: 
# 
# 1. A parquet file that provides, for a given test, all of the other tests that are highly correlated (correlation coefficient of 0.9 or above). This file omits any tests that do not have any highly correlated tests. So, if a test is not present on the list, then it has no highly correlated tests associated with it at this time and has been removed from the record. The calculation for correlation is performed on all available data exposed by the Red Hat test grid instance at the time the notebook is run.
# 
# 2. A summary metric that can be easily tracked over time that represents the average size of correlated test sets in the above parquet. 
# 
# 
# __Note__: This notebook follows a very similar approach to an earlier [EDA notebook](https://github.com/aicoe-aiops/ocp-ci-analysis/blob/master/notebooks/data-sources/Sippy/sippy_failure_correlation.ipynb) where we correlated failures with a different dataset. For simplicity, much of the reasoning behind the decisions made in this notebook have been omited here, but can be found in the above linked notebook for interested readers :)   
# 
# 
# [related  issue #139](https://github.com/aicoe-aiops/ocp-ci-analysis/issues/139)

# In[1]:


# Import libraries
import gzip
import json
import os
import numpy as np
import pandas as pd
import datetime

from ipynb.fs.defs.metric_template import decode_run_length
from ipynb.fs.defs.metric_template import CephCommunication
from ipynb.fs.defs.metric_template import save_to_disk, read_from_disk
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


# In[2]:


## Specify variables

METRIC_NAME = "correlation"

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


# # Calculation 
# 
# Here we iterate through each grid in our dataset and collect the the names of all the tests that fail during the same build. We will store this in the `failure_groups` list.

# In[4]:


failure_groups = []

for tab in list(testgrid_data.keys()):
    for grid in testgrid_data[tab].keys():
        current_grid = testgrid_data[tab][grid]

        tests = [
            current_grid["grid"][i]["name"] for i in range(len(current_grid["grid"]))
        ]
        # unroll the run-length encoding and set bool for flake or not (x==13)
        decoded = [
            (
                np.array(decode_run_length(current_grid["grid"][i]["statuses"])) == 12
            ).tolist()
            for i in range(len(current_grid["grid"]))
        ]

        matrix = pd.DataFrame(zip(tests, decoded), columns=["test", "values"])
        matrix = pd.DataFrame(matrix["values"].to_list(), index=matrix["test"])

        for c, items in matrix.iteritems():
            if len(items[items].index) > 1:
                failure_groups.append(items[items].index)


# In[5]:


failure_groups = pd.Series(failure_groups)


# In[6]:


len(failure_groups)


# Now we want to define a vocabulary for all of the unique tests in our dataset so that we can encode our failure sets using a binary encoding scheme.

# In[7]:


vocab = set()
count = 0
for fg in failure_groups:
    count += len(fg)
    vocab.update(fg)

vocab = list(vocab)
print(count)
len(vocab)


# Confirm that there are no duplicates in the vocab to ensure we have a unique set

# In[8]:


len(pd.Series(vocab).unique()) == len(vocab)


# Now we'll use the below function to create our binary encoded vectors for our correlation analysis

# In[9]:


def encode_tests(job):
    encoded = []
    for v in vocab:
        if v in job:
            encoded.extend([1])
        else:
            encoded.extend([0])
    return encoded


# In[10]:


encoded = failure_groups.apply(encode_tests)


# In[11]:


encoded.head()


# In[12]:


df_encoded = pd.DataFrame(encoded.array, columns=vocab)
df_encoded.head()


# In[13]:


# percent that each test is present in the data; percent failure
perc_present = df_encoded.sum() / len(df_encoded)
perc_present.sort_values(ascending=False).head(3)


# In[14]:


# Total failure count present in the data; failure per test
occurrence_count = df_encoded.sum()
occurrence_count.sort_values(ascending=False).head(3)


# We also want to make sure that our correlation values are not just due to unique failed test sets present in our dataset. We want to make sure our tests impact multiple jobs. For example, if we had a unique failed test set that only occurred in a single example, and shared no other failed tests among the vocabulary, then all of the tests would appear to be 100% correlated with each other, when in fact that is merely a consequence of insufficient data. In order to prevent that, we will ignore any tests that occur only in a single job. In order to do that we will use occurrence_count to create a filter vector for any test that occurs only once. Then filter them out of our working DF.

# In[15]:


filter_unique = list(occurrence_count[occurrence_count.values <= 1].index)


# In[16]:


df_encoded = df_encoded.drop(filter_unique, axis=1)


# In[17]:


df_encoded.shape


# In[18]:


# this takes time with full dataset - ~ 2 hours may need to use different approach
# todo try with dask
corr_matrix = df_encoded.corr()


# In[20]:


# For each feature, find the other features that are correlated by more than 0.9
top_correlation = {}

for c in corr_matrix.columns:
    top_correlation[c] = []
    series = corr_matrix.loc[c]

    for i, s in enumerate(series):
        if s > 0.90 and series.index[i] != c:
            top_correlation[c].append((series.index[i], s))

len(top_correlation)


# # Examine example output
# 
# Let's go ahead and take a look at which tests are highly correlated with the first test in our results list.

# In[24]:


# top_correlation has a number of empty rows as not all tests have high correlations with others,
# lets grab only the sets that have at least 1 highly correlated test

pd.set_option("display.max_colwidth", 150)
corr_sets = []
for i in top_correlation.items():
    if len(i[1]) >= 1:
        corr_sets.append(i)
print(f"{len(corr_sets)} sets of correlated tests \n")
print(f"Feature of interest: {corr_sets[1][0]}")
pd.DataFrame(corr_sets[1][1], columns=["test_name", "correlation coefficient"])


# In[25]:


if not AUTOMATION:
    test_name = "openshift-tests.[k8s.io] Security Context When creating a container with runAsUser should run the container with uid 65534 [LinuxOnly] [NodeConformance] [Conformance] [Suite:openshift/conformance/parallel/minimal] [Suite:k8s]"  # noqa
    num = occurrence_count.loc[test_name]
    print(f"{num} : the number of times this test failed in our data set")


# In[26]:


lst = []
focus = corr_sets[1][1]
for j in focus:
    lst.append((j[0], occurrence_count.loc[j[0]]))

pd.DataFrame(lst, columns=["test_name", "num_occurrences"])


# ### Save to Ceph or local

# In[27]:


save = pd.DataFrame(corr_sets, columns=["test_name", "correlated_tests"])
save["correlated_tests"] = save["correlated_tests"].apply(str)

if AUTOMATION:
    cc = CephCommunication(s3_endpoint_url, s3_access_key, s3_secret_key, s3_bucket)
    cc.upload_to_ceph(
        save,
        s3_path,
        f"{METRIC_NAME}/{METRIC_NAME}-{timestamp.year}-{timestamp.month}-{timestamp.day}.parquet",
    )
else:
    save_to_disk(
        save,
        OUTPUT_DATA_PATH,
        f"{METRIC_NAME}-{timestamp.year}-{timestamp.month}-{timestamp.day}.parquet",
    )


# In[28]:


## Sanity check to see if the dataset is the same
if AUTOMATION:
    sanity_check = cc.read_from_ceph(
        s3_path,
        f"{METRIC_NAME}/{METRIC_NAME}-{timestamp.year}-{timestamp.month}-{timestamp.day}.parquet",
    ).head()
else:
    sanity_check = read_from_disk(
        OUTPUT_DATA_PATH,
        f"{METRIC_NAME}-{timestamp.year}-{timestamp.month}-{timestamp.day}.parquet",
    ).head()

sanity_check


# #### Lets also capture the average size of correlated failure groups to track over time 

# In[29]:


average_corr = save["correlated_tests"].apply(len).mean()
metric_to_save = pd.DataFrame(
    [[timestamp, average_corr]],
    columns=["timestamp", "average_number_of_correlated_failures"],
)


if AUTOMATION:
    cc.upload_to_ceph(
        metric_to_save,
        s3_path,
        f"avg_{METRIC_NAME}/avg_{METRIC_NAME}-{timestamp.year}-{timestamp.month}-{timestamp.day}.parquet",
    )
else:
    save_to_disk(
        metric_to_save,
        OUTPUT_DATA_PATH,
        f"avg_{METRIC_NAME}-{timestamp.year}-{timestamp.month}-{timestamp.day}.parquet",
    )


# In[30]:


## Sanity check to see if the dataset is the same

if AUTOMATION:
    sanity_check = cc.read_from_ceph(
        s3_path,
        f"avg_{METRIC_NAME}/avg_{METRIC_NAME}-{timestamp.year}-{timestamp.month}-{timestamp.day}.parquet",
    ).head()
else:
    sanity_check = read_from_disk(
        OUTPUT_DATA_PATH,
        f"avg_{METRIC_NAME}-{timestamp.year}-{timestamp.month}-{timestamp.day}.parquet",
    ).head()

sanity_check


# ### Conclusion
# 
# This notebook collected all sets of highly correlated tests, i.e, sets of tests that most commonly fail together and stored that data in ceph as well as locally. A user can now pull this data and, given a test name of interest, be provided a list of all other highly correlated tests. 
# 
# 
# This notebook also computed a numerical value to summarize and quantify these correlations in aggregate: the average size of failure correlation sets. This value is also stored both locally and in ceph. 
