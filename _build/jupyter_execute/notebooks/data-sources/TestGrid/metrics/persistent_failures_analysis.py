#!/usr/bin/env python
# coding: utf-8

# # Persistent Failures Analysis
# 
# Speed and quality of builds are some of the key performance indicators for the continuous integration process. That is, reduction in the number of failing builds, or the time taken to fix them, should generally indicate an improvement in the development process. In this notebook, we will analyze the data collected from [TestGrid](https://testgrid.k8s.io/redhat) to calculate metrics such as percent of failures that persist for long times, how long do such failures last for (i.e. how long does it take to fix them), etc. Our goal here is to provide engineers and managers with insights such as:
# 
# - Which tests (e.g. network or storage) have the most "long lasting" failures
# - Which platforms (e.g. AWS or bare metal) have the most "long lasting" failures
# - How long does it take to get a failing test passing again
# - How long does it take to get a failing build to build again
# 
# In this notebook, we will follow the same convention as in [number_of_flakes.ipynb](number_of_flakes.ipynb), i.e., create a long dataframe and let the end user decide what level they want to create aggregate views at.
# 
# Linked issue: [issue](https://github.com/aicoe-aiops/ocp-ci-analysis/issues/116)

# In[1]:


import gzip
import json

import pandas as pd
import os
import datetime

from ipynb.fs.defs.metric_template import CephCommunication
from ipynb.fs.defs.metric_template import save_to_disk, read_from_disk
from ipynb.fs.defs.metric_template import TestStatus
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


# In[2]:


## Specify variables

METRIC_NAME = "persistent_failures"

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
# 
# To find persistent failures, we find the number of failing tests. Once all failing-status tests have been found, we determine if they are one-time failures or consecutive failures by looking at the status of the test at the next time stamp. We'll also measure the length of time it takes for failing tests to pass so we can further investigate time-to-fix metrics.

# In[4]:


# calculate consecutive failure stats
consec_fail_stats_tuples = []

for tab in testgrid_data.keys():
    print(tab)

    for grid in testgrid_data[tab].keys():
        current_grid = testgrid_data[tab][grid]

        ## Extract relevant info for each test
        for current_test in current_grid["grid"]:

            # number of failing cells
            n_failing_cells = 0

            # total number of occurences of failures (consecutive or one-time)
            n_fail_instances = 0

            # number of occurences of consecutive (not "one-time") failures
            n_consecutive_fail_instances = 0

            # times spent fixing each occurence of failure
            times_spent = []

            # helper variables for calculating time spent fixing
            prev_failing = False
            prev_passing = False
            curr_time_spent = 0
            prev_oldest_ts_idx = 0

            # prev_not_passing = False

            # total number of timestamps
            n_cells = 0

            # total number of transitions form pass to fail
            # and fail to pass
            pass_to_fail = 0
            fail_to_pass = 0

            for e, s in enumerate(current_test["statuses"]):
                # oldest (least recent) timestamp in current rle encoded dict
                curr_oldest_ts_idx = prev_oldest_ts_idx + s["count"]

                # if the current status is not failing and the previous (i.e.
                # the "newer") status was failing, then this marks the start
                # point of failure. since end point would have already been
                # calculated in previous loop, we just need to save time spent
                if s["value"] != TestStatus.FAIL.value:
                    if prev_failing:
                        times_spent.append(curr_time_spent)
                        curr_time_spent = 0
                        if s["value"] == TestStatus.PASS.value:
                            pass_to_fail += (
                                1  # Inverted because the dictioary is from most recent
                            )

                elif s["value"] == TestStatus.FAIL.value:
                    n_fail_instances += 1
                    n_failing_cells += s["count"]
                    if s["count"] > 1:
                        n_consecutive_fail_instances += 1
                    if prev_passing:
                        fail_to_pass += (
                            1  # Inverted because the dictioary is from most recent
                        )

                    # if previous (i.e. the "newer") status was not failing
                    # and now its failing, then time delta between the oldest
                    # ts from previous status and current one must have been
                    # spent fixing the failure
                    if not prev_failing:
                        curr_time_spent += (
                            current_grid["timestamps"][max(0, prev_oldest_ts_idx - 1)]
                            - current_grid["timestamps"][curr_oldest_ts_idx - 1]
                        )

                # update helper variables
                prev_failing = s["value"] == TestStatus.FAIL.value
                prev_passing = s["value"] == TestStatus.PASS.value
                prev_oldest_ts_idx = curr_oldest_ts_idx
                n_cells += s["count"]

            # test never got to non-fail status again so time spent so far
            # wont have been added to times_spent yet
            if curr_time_spent != 0:
                times_spent.append(curr_time_spent)

            ## Calculate stats for this test

            # consecutive failure rate
            try:
                consec_fail_rate = n_consecutive_fail_instances / n_fail_instances
            except ZeroDivisionError:
                consec_fail_rate = 0

            # mean length of failures
            try:
                mean_fail_len = n_failing_cells / n_fail_instances
            except ZeroDivisionError:
                mean_fail_len = 0

            # mean time to fix
            try:
                mean_time_to_fix = sum(times_spent) / len(times_spent)
            except ZeroDivisionError:
                mean_time_to_fix = 0

            pass_to_fail_rate = pass_to_fail / n_cells
            fail_to_pass_rate = fail_to_pass / n_cells

            # save the results to list
            consec_fail_stats_tuples.append(
                [
                    tab,
                    grid,
                    current_test["name"],
                    consec_fail_rate,
                    mean_fail_len,
                    mean_time_to_fix,
                    pass_to_fail_rate,
                    fail_to_pass_rate,
                ]
            )

len(consec_fail_stats_tuples)


# In[5]:


# put results in a pretty dataframe
consec_fail_stats_df = pd.DataFrame(
    data=consec_fail_stats_tuples,
    columns=[
        "tab",
        "grid",
        "test",
        "consec_fail_rate",
        "mean_fail_len",
        "mean_time_to_fix",
        "pass_to_fail_rate",
        "fail_to_pass_rate",
    ],
)
consec_fail_stats_df.head()


# In[6]:


# the output here shows what tabs and grids have overall the longest failures
res = consec_fail_stats_df[consec_fail_stats_df["test"].str.contains("Overall")]
res.sort_values("mean_time_to_fix", ascending=False).head()


# ## Save to Ceph or Local

# In[7]:


if AUTOMATION:
    cc = CephCommunication(s3_endpoint_url, s3_access_key, s3_secret_key, s3_bucket)
    cc.upload_to_ceph(
        consec_fail_stats_df,
        s3_path,
        f"{METRIC_NAME}/{METRIC_NAME}-{timestamp.year}-{timestamp.month}-{timestamp.day}.parquet",
    )
else:
    save_to_disk(
        consec_fail_stats_df,
        OUTPUT_DATA_PATH,
        f"{METRIC_NAME}-{timestamp.year}-{timestamp.month}-{timestamp.day}.parquet",
    )


# In[8]:


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
# This notebook computed the mean fail length, the mean time to fix failures, pass-to-fail rate, and fail-to-pass rate for tests. The dataframe saved on ceph can be used to generate aggregated views and visualizations.
