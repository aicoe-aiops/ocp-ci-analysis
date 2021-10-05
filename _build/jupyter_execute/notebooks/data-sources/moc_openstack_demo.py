#!/usr/bin/env python
# coding: utf-8

# # MOC OpenStack S3 Demo

# In this notebook we will give a quick demo on how reading, writing and storing data works in MOC OpenStack.
# 
# Namely, we will show that you do not need credentials to read data, but you do need credentials to write and upload data.

# In[1]:


import boto3
from dotenv import find_dotenv, load_dotenv
import os
from botocore import UNSIGNED
from botocore.client import Config


# In[2]:


load_dotenv(find_dotenv())


# In[3]:


# this url and bucket are public knowledge

s3_endpoint_url = os.getenv("S3_ENDPOINT", "https://kzn-swift.massopen.cloud:443")
s3_bucket = os.getenv("S3_BUCKET", "ai4ci")


# In[4]:


s3_access_key = os.getenv("S3_ACCESS_KEY")
s3_secret_key = os.getenv("S3_SECRET_KEY")


# First we make a little test file.

# In[5]:


with open("test.txt", "w") as f:
    f.write("test text (:")


# In[6]:


get_ipython().system(' cat test.txt')


# We set up two connections, one with and one without credentials.

# In[7]:


# note if using boto3 anonymously you must add this config argument

unauth_connection = boto3.client(
    "s3", endpoint_url=s3_endpoint_url, config=Config(signature_version=UNSIGNED)
)


# In[8]:


auth_connection = boto3.client(
    "s3",
    aws_access_key_id=s3_access_key,
    aws_secret_access_key=s3_secret_key,
    endpoint_url=s3_endpoint_url,
)


# We try to upload our test file using the unauthenticated connection.

# In[9]:


try:
    unauth_connection.upload_file(Bucket=s3_bucket, Key="test.txt", Filename="test.txt")
    print("upload successful")
except Exception as e:
    print(e)


# Now we do the same for the authenticated connection

# In[10]:


try:
    auth_connection.upload_file(Bucket=s3_bucket, Key="test.txt", Filename="test.txt")
    print("upload successful")
except Exception as e:
    print(e)


# Now that we know the file is uploaded, we try to read it using the unauthenticated connection.

# In[11]:


response = unauth_connection.get_object(Bucket=s3_bucket, Key="test.txt")

for i in response["Body"]:
    print(i.decode())


# Now we read it with the authenticated connection.

# In[12]:


response = auth_connection.get_object(Bucket=s3_bucket, Key="test.txt")

for i in response["Body"]:
    print(i.decode())


# Now we try to delete the file, with the unauthenticated connection and then with the authenticated connection.

# In[13]:


try:
    unauth_connection.delete_object(Bucket=s3_bucket, Key="test.txt")
    print("unauth delete successful")
except Exception as e:
    print(e)
    print("unauth delete unsuccessful")

print("")

try:
    auth_connection.delete_object(Bucket=s3_bucket, Key="test.txt")
    print("auth delete successful")
except Exception as e:
    print(e)
    print("auth delete unsuccessful")


# In[14]:


# check that it is indeed deleted
try:
    response = auth_connection.get_object(Bucket=s3_bucket, Key="test.txt")
except Exception as e:
    print(e)


# In[15]:


os.remove("test.txt")

