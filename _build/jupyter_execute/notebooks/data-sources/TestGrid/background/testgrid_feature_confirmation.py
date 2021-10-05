#!/usr/bin/env python
# coding: utf-8

# # TestGrid Additional Features - uniform or unique?
# 
# As can be seen in an [earlier notebook](../testgrid_EDA.ipynb), TestGrids have a more metadata (features) than just the test values we've been focused on. 
# 
# In this notebook we are going to take a closer look at these other metadata fields for both Openshift and Kubernetes and determine if they are uniform across grids and worth taking a closer look at our are distinct by grid. 

# In[1]:


import requests
from bs4 import BeautifulSoup


# In[2]:


# access the testgrid.k8s.io to get the dashboards for Red Hat
response = requests.get(
    "https://testgrid.k8s.io/redhat-openshift-informing?id=dashboard-group-bar"
)


# In[3]:


html = BeautifulSoup(response.content)
testgrid_script = html.findAll("script")[3]
testgrid_script = testgrid_script.text.split()[5].split(",")
dashboard_names = [x.split(":")[1] for x in testgrid_script if "name" in x]


# In[4]:


# Print all the feature names for one grid
response = requests.get(
    "https://testgrid.k8s.io/redhat-openshift-ocp-release-4.2-informing/table? \
    &show-stale-tests=&tab=release-openshift-origin-installer-e2e-aws-upgrade-rollback-4.1-to-4.2"
)
for i in response.json().keys():
    print(i)


# This is a list for one specific dashboard, we need to check and make sure that this is standard across all Red Hat and Kubernetes grids before moving forward. 

# In[5]:


# Iterate through each board and collect
download = True
if download is True:
    available_features = {}

    for dashboard in dashboard_names:
        response_1 = requests.get(f"https://testgrid.k8s.io/{dashboard}/summary")
        jobs = response_1.json().keys()
        for job in jobs:
            response_2 = requests.get(
                f"https://testgrid.k8s.io/{dashboard}/table?&show-stale-tests=&tab={job}&grid=old"
            )
            if response_2.status_code != 200:
                continue

            available_features[f"{dashboard}_{job}"] = response_2.json().keys()
        print(f"{dashboard} downloaded ")
else:
    print("Not Downloading")


# In[6]:


### for each grid this will print out any features they are missing from a master list
all_features = []
for i in available_features.keys():
    all_features.extend(list(available_features[i]))

uniqe_list_redhat = set(all_features)

for i in available_features.keys():
    print(i)
    print(uniqe_list_redhat.difference(set(available_features[i])))


# We can see above that all the features are the same for each grid expect most do not include the feature "alerts"

# ## Kubernetes Feature Set

# In[7]:


dashboard = '"google-aws"'
job = "kops-aws-cni-amazon-vpc"
response = requests.get("https://testgrid.k8s.io/google-aws?id=dashboard-group-bar")
response_2 = requests.get(
    f"https://testgrid.k8s.io/{dashboard}/table?&show-stale-tests=&tab={job}"
)


# In[8]:


# get the grid names for google-aws
html = BeautifulSoup(response.content)
testgrid_script = html.findAll("script")[3]
testgrid_script = testgrid_script.text.split()[5].split(",")
dashboard_names = [x.split(":")[1] for x in testgrid_script if "name" in x]
dashboard_names


# In[9]:


# Print all the feature names for one grid
k8s_features = list(response_2.json().keys())
for i in response_2.json().keys():
    print(i)


# In[10]:


# compare this grid with the Red Hat master list of features
uniqe_list_redhat.difference(set(k8s_features))


# again, we can see that in this case the Kubernetes and Red Hat grids appear to be the same except for "alerts" in some cases.
# 
# Let's double check an iterate through all of the google grids. 

# In[11]:


download = True
if download is True:
    available_features = {}

    for dashboard in dashboard_names:
        response_1 = requests.get(f"https://testgrid.k8s.io/{dashboard}/summary")
        jobs = response_1.json().keys()
        for job in jobs:
            response_2 = requests.get(
                f"https://testgrid.k8s.io/{dashboard}/table?&show-stale-tests=&tab={job}"
            )
            if response_2.status_code != 200:
                continue

            available_features[f"{dashboard}_{job}"] = response_2.json().keys()

        print(f"{dashboard} downloaded ")
else:
    print("Not Downloading")

all_features = []
for i in available_features.keys():
    all_features.extend(list(available_features[i]))

uniqe_list_k8s = set(all_features)


# In[12]:


for i in available_features.keys():
    print(uniqe_list_k8s.difference(set(available_features[i])))


# In[13]:


uniqe_list_k8s.difference(uniqe_list_redhat)


# Great so, we can see from the above that not only do we have the same situation for the google-aws set of grids, but there is also no difference between the full Red Hat and Kubernetes lists of features.
# 
# Therefore we know that we have a reliable set of features that we can look at beyond the grid tests themselves. 
