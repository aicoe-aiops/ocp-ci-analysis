#!/usr/bin/env python
# coding: utf-8

# # Percent of Failing Tests Fixed
# 
# This notebook is an addition to the series of KPI notebook in which we calculate key performance indicators for CI processes. In this notebook, we will calculate the KPI "Percent of failing tests fixed in each run/timestamp." Essentially, we will determine 
# 
# - percent of tests that were failing and are now fixed
# 
# For OpenShift managers, this information can potentially help quantify the agility and efficiency of their team. If this number is high, it means they are able to quickly identify the root causes of all failing tests in the previous run and fix them. Conversely if this number is low, it means only a small percent of previously failing tests get fixed in each new run, which in turn implies that their CI process is likely not as efficient as it could be.
# 
# Related issues: [#149](https://github.com/aicoe-aiops/ocp-ci-analysis/issues/149)

# In[1]:


import os
import gzip
import json
import datetime

import numpy as np
import pandas as pd

from ipynb.fs.defs.metric_template import decode_run_length
from ipynb.fs.defs.metric_template import testgrid_labelwise_encoding
from ipynb.fs.defs.metric_template import CephCommunication
from ipynb.fs.defs.metric_template import save_to_disk, read_from_disk

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


# In[2]:


## Specify variables

METRIC_NAME = "pct_fixed_each_ts"

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


# ## Calculation
# 
# To find fixed tests, we modified the [testgrid_labelwise_encoding function](./number_of_flakes.ipynb). The loop is adapted to put a "True" if a test was fixed in the current run, and "False" otherwise. Basically instead of indicating "is_flake" or "is_pass," it indicates "is passing now but was failing before" aka "is_flip."

# In[4]:


# NOTE: this for loop is a modified version of the testgrid_labelwise_encoding function

percent_label_by_grid_csv = []

for tab in testgrid_data.keys():
    print(tab)

    for grid in testgrid_data[tab].keys():
        current_grid = testgrid_data[tab][grid]

        # get all timestamps for this grid (x-axis of grid)
        timestamps = [
            datetime.datetime.fromtimestamp(t // 1000)
            for t in current_grid["timestamps"]
        ]

        tests = []
        all_tests_did_get_fixed = []

        # NOTE: this list of dicts goes from most recent to least recent
        for i, current_test in enumerate(current_grid["grid"]):
            tests.append(current_test["name"])
            statuses_decoded = decode_run_length(current_grid["grid"][i]["statuses"])

            did_get_fixed = []
            for status_i in range(0, len(statuses_decoded) - 1):
                did_get_fixed.append(
                    statuses_decoded[status_i] == 1
                    and statuses_decoded[status_i + 1] == 12
                )

            # the least recent test cannot have "True", assuming it wasnt failing before
            did_get_fixed.append(False)

            # add results for all timestamps for current test
            all_tests_did_get_fixed.append(np.array(did_get_fixed))

        all_tests_did_get_fixed = [
            list(zip(timestamps, g)) for g in all_tests_did_get_fixed
        ]

        # add the test, tab and grid name to each entry
        # TODO: any ideas for avoiding this quad-loop
        for i, d in enumerate(all_tests_did_get_fixed):
            for j, k in enumerate(d):
                all_tests_did_get_fixed[i][j] = (k[0], tab, grid, tests[i], k[1])

        # accumulate the results
        percent_label_by_grid_csv.append(all_tests_did_get_fixed)

# output above leaves us with a doubly nested list. Flatten
flat_list = [item for sublist in percent_label_by_grid_csv for item in sublist]
flatter_list = [item for sublist in flat_list for item in sublist]


# In[5]:


flatter_list[0]


# In[6]:


# this df indicates whether a test was fixed or not at a given timestamp (as compared to previous one)
df_csv = pd.DataFrame(
    flatter_list, columns=["timestamp", "tab", "grid", "test", "did_get_fixed"]
)
df_csv.head()


# In[7]:


# each element in this multiindexed series tells how many tests got fixed at each run/timestamp
num_fixed_per_ts = df_csv.groupby(["tab", "grid", "timestamp"]).did_get_fixed.sum()
num_fixed_per_ts


# In[8]:


build_failures_list = testgrid_labelwise_encoding(testgrid_data, 12)


# In[9]:


# this df indicates whether a test was failing or not at a given timestamp
failures_df = pd.DataFrame(
    build_failures_list,
    columns=["timestamp", "tab", "grid", "test", "test_duration", "failure"],
)
failures_df.head()


# In[10]:


# each element in this multiindexed series tells how many tests failed at each run/timestamp
num_failures_per_ts = failures_df.groupby(["tab", "grid", "timestamp"]).failure.sum()
num_failures_per_ts


# In[11]:


# dividing the above two df's tells us what percent of failing tests got fixed at each timestamp
pct_fixed_per_ts = (num_fixed_per_ts / num_failures_per_ts.shift()).fillna(0)
pct_fixed_per_ts


# In[12]:


# convert to df from multiindex series
pct_fixed_per_ts_df = pct_fixed_per_ts.reset_index().rename(columns={0: "pct_fixed"})
pct_fixed_per_ts_df


# ## Save to Ceph or local
# Save the data frame in a parquet format on the Ceph bucket or locally

# In[13]:


save = pct_fixed_per_ts_df

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


# In[14]:


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


# ## Conclusion
# This notebook computed the mean fail length, the mean time to fix failures, pass-to-fail rate, and fail-to-pass rate for tests. The dataframe saved on ceph can be used to generate aggregated views and visualizations on the percent of fixed tests at each timestamp.
