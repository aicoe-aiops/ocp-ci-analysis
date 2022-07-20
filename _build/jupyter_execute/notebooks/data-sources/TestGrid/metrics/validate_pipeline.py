#!/usr/bin/env python
# coding: utf-8

# # Validate the succesful running of the Automated Pipeline

# Successful running of a pipeline should collected raw data and metrics in the project S3 bucket.
# This notebook is a simple utlity notebook to check the contents of the S3 bucket and validate that the automated pipeline ran succesfully.

# In[10]:


import os
import datetime
from dotenv import load_dotenv, find_dotenv
from ipynb.fs.defs.metric_template import CephCommunication


# In[11]:


load_dotenv(find_dotenv())


# In[12]:


s3_endpoint_url = os.getenv("S3_ENDPOINT")
s3_access_key = os.getenv("S3_ACCESS_KEY")
s3_secret_key = os.getenv("S3_SECRET_KEY")
s3_bucket = os.getenv("S3_BUCKET")


# In[13]:


s3_bucket


# In[14]:


cc = CephCommunication(s3_endpoint_url, s3_access_key, s3_secret_key, s3_bucket)


# In[15]:


bucket = cc.s3_resource.Bucket(s3_bucket)


# In[16]:


# check all .parquet files in S3
objects = [i for i in bucket.objects.all() if "parquet" in i.key]
sorted(objects, key=lambda x: x.key)


# In[17]:


# check all raw data files in s3
objects = [i for i in bucket.objects.all() if "raw_data/" in i.key]
sorted(objects, key=lambda x: x.key)


# ### Check for today's data on S3

# In[18]:


# modify variables for custom date
timestamp = datetime.datetime.today()


# In[19]:


metric_name = f"{timestamp.year}-{timestamp.month}-{timestamp.day}.parquet"
raw_data = f"testgrid_{timestamp.day}{timestamp.month}.json"


# In[20]:


metric_objects = [i for i in bucket.objects.all() if metric_name in i.key]
sorted(metric_objects, key=lambda x: x.key)


# In[21]:


raw_data_objects = [i for i in bucket.objects.all() if raw_data in i.key]
sorted(raw_data_objects, key=lambda x: x.key)

