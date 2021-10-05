#!/usr/bin/env python
# coding: utf-8

# # OpenShift Origin PR Data Extraction
# 
# In this notebook, we will collect the raw Pull Request data from the [OpenShift Origin](https://github.com/openshift/origin) github repo, and save it as a json file on an s3 bucket. To do this, we will use the `srcopsmetrics` tool developed by the Thoth team.

# In[1]:


import os
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())


# In[2]:


# get the org/repo from env vars
ORG = os.environ["GITHUB_ORG"]
REPO = os.environ["GITHUB_REPO"]

print(f"{ORG}/{REPO}")


# In[ ]:


# run collection on the org/repo specified
get_ipython().system('python -m srcopsmetrics.cli --create-knowledge --repository $ORG/$REPO --entities PullRequest')


# # Conclusion
# 
# By running this notebook we have collected and stored the github PR data to our s3 bucket. It is now ready for the cleaning and feature engineering steps of the ML workflow.
