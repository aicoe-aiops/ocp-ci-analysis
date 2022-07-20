#!/usr/bin/env python
# coding: utf-8

# # Fetch test grid data
# * In this notebook, we fetch relevant data from https://testgrid.k8s.io and save on Ceph for further analysis.
# * This is supposed to be run in automation as a part of kubeflow pipeline.

# In[1]:


## Import libraries
import datetime
import os
import json
from dotenv import load_dotenv, find_dotenv
import requests
from bs4 import BeautifulSoup
from ipynb.fs.defs.metric_template import CephCommunication

load_dotenv(find_dotenv())


# In[2]:


## Specify variables

# Specify the path for input grid data
INPUT_DATA_PATH = (
    "https://testgrid.k8s.io/redhat-openshift-informing?id=dashboard-group-bar"
)

# Specify the path for output raw data
OUTPUT_DATA_PATH = "../../../../data/raw"

## CEPH Bucket variables
## Create a .env file on your local with the correct configs
s3_endpoint_url = os.getenv("S3_ENDPOINT")
s3_access_key = os.getenv("S3_ACCESS_KEY")
s3_secret_key = os.getenv("S3_SECRET_KEY")
s3_bucket = os.getenv("S3_BUCKET")
s3_input_data_path = "raw_data"

# Specify whether or not we are running this as a notebook or part of an automation pipeline.
AUTOMATION = os.getenv("IN_AUTOMATION")


# In[3]:


## Connect to the url and fetch dashboard names
response = requests.get(INPUT_DATA_PATH)
html = BeautifulSoup(response.content)
testgrid_script = html.findAll("script")[3]
testgrid_script = testgrid_script.text.split()[5].split(",")
dashboard_names = [x.split(":")[1] for x in testgrid_script if "name" in x]
dashboard_names


# In[4]:


## Download the dashboard data
download = True
if download:
    data_set = {}

    for dashboard in dashboard_names:
        response_1 = requests.get(f"https://testgrid.k8s.io/{dashboard}/summary")
        jobs = response_1.json().keys()
        dashboard_jobs = {}

        for job in jobs:
            response_2 = requests.get(
                f"https://testgrid.k8s.io/{dashboard}/table?&show-stale-tests=&tab={job}&graph-metrics=test-duration-minutes"  # noqa
            )
            if response_2.status_code != 200:
                continue

            if "tests" in response_2.json():
                grid = []
                for x in response_2.json()["tests"]:
                    test = {"name": x["name"], "statuses": x["statuses"]}
                    if "graphs" in x.keys():
                        test["graphs"] = x["graphs"]
                    else:
                        test["graphs"] = None
                    grid.append(test)

                time_stamps = response_2.json()["timestamps"]

                dashboard_jobs[job] = {"grid": grid, "timestamps": time_stamps}

        data_set[dashboard] = dashboard_jobs
        print(f"{dashboard} downloaded ")
else:
    print("Not Downloading")


# In[5]:


## Set filename
date = datetime.datetime.today()
filename = f"testgrid_{date.day}{date.month}.json"


# In[6]:


timestamp = datetime.datetime.now()

if AUTOMATION:
    ## Connect to Ceph
    cc = CephCommunication(s3_endpoint_url, s3_access_key, s3_secret_key, s3_bucket)

    ## Put data on ceph
    s3_obj = cc.s3_resource.Object(s3_bucket, f"{s3_input_data_path}/{filename}")
    status = s3_obj.put(Body=bytes(json.dumps(data_set).encode("UTF-8")))

    ## Print Status
    print(status)

else:
    file_path = f"{OUTPUT_DATA_PATH}/{filename}"
    with open(file_path, "w") as outfile:
        json.dump(data_set, outfile)

