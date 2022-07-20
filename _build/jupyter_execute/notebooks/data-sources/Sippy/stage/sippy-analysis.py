#!/usr/bin/env python
# coding: utf-8

# # SIPPY 
# A tool to process the job results from https://testgrid.k8s.io/
# 
# Sippy provides dashboards  for the openshift CI test/job data.
# It contains the health summary for CI’s latest release.
# Reports on which tests fail most frequently along different dimensions:
# * overall
# * by job
# * by platform (e.g. aws, gcp, etc)
# * by sig (sig ownership of the test)
# * Job runs that had large groups of test failures in a single run (generally indicative of a fundamental issue rather than a test problem)
# * Job pass rates (which jobs are failing frequently, which are not, in sorted order)
# 
# In this notebook we will be looking at the existing testgrid data at testgrid.k8s.io, giving specific attention to [Red Hat's CI dashboards](https://testgrid.k8s.io/redhat-openshift-informing).
# 

# ###  Getting the data :

# In[1]:


import requests
import os
import sys
import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# adding path to notebook consisting of modules to be imported
module_path = os.path.abspath(os.path.join("../../TestGrid"))
if module_path not in sys.path:
    sys.path.append(module_path)


# In[3]:


from ipynb.fs.defs.testgrid_EDA import decode_run_length  # noqa: E402


# ### List all the available dashboards
# 
# We need a programmatic way to access all the available Red Hat Dashboards on testgrid. This can be done by scrapping the html of any one of the dashboards. 

# In[4]:


response = requests.get(
    "https://testgrid.k8s.io/redhat-openshift-informing?id=dashboard-group-bar"
)


# In[5]:


html = BeautifulSoup(response.content)
testgrid_script = html.findAll("script")[3]
testgrid_script = testgrid_script.text.split()[5].split(",")
dashboard_names = [x.split(":")[1] for x in testgrid_script if "name" in x]
dashboard_names


# ### Inspect at a specific release summary dashboard
# 
# Now that we have a programmatic way of collecting all the dashboard names we can use this later on to collect all the available data provided by testgrid. Let's take a look at a specific dashboard and see what kind of info its summary holds. 

# In[6]:


dashboard = "redhat-openshift-ocp-release-4.6-informing"


# In[7]:


response = requests.get(f"https://testgrid.k8s.io/{dashboard}/summary")


# In[8]:


data = response.json()


# In[9]:


ocp46 = pd.DataFrame(data)
ocp46.columns


# For each dashboard there are a number of jobs associated with it (We will use these column names later to get the access the detailed data). And for each job we have a number of features, some which contain additional data. 
# 
# * last run
# * last update
# * latest_green
# * overall_status
# * overall_status_icon
# * status**
# * tests**
# * dashboard_name
# * healthiness**
# 
# _** features that have additional data_

# In[10]:


print(ocp46.columns[0])
ocp46.loc["tests", ocp46.columns[0]]


# For a detailed analysis, we will be looking into the data for a particular job from the (shown in the list above) 
# * dashboard : `redhat-openshift-ocp-release-4.6-informing`  
# * job : `https://testgrid.k8s.io/redhat-openshift-ocp-release-4.2-informing/table?&show-stale-tests=&tab=release-openshift-origin-installer-e2e-aws-shared-vpc-4.6`   
# 

# In[11]:


response = requests.get(
    "https://testgrid.k8s.io/redhat-openshift-ocp-release-4.6-informing/table? \
&show-stale-tests=&tab=release-openshift-origin-installer-e2e-aws-shared-vpc-4.6"
)  # noqa


# In[12]:


for i in response.json().keys():
    print(i)


# In[13]:


details = pd.DataFrame(response.json()["tests"])
details.columns


# In[14]:


details.head()


# In[15]:


target = details.target.unique()


# In[16]:


messages = details.messages
messages.to_frame()


# In[17]:


test_names = details["name"].unique()
test_names = pd.DataFrame(test_names)
test_names = test_names.rename(columns={0: "Test Name"})
with pd.option_context("display.max_rows", None, "display.max_columns", None):
    display(test_names[:10])


# In[18]:


details.name


# Split up the value at name to get exact test name. And store these values in two separate columns : Type (first bit) and Test Name (second bit)

# In[19]:


new = details["name"].str.split(".", n=1, expand=True)
details["Type"] = new[0]
details["Test Name"] = new[1]
details.drop(columns=["name"], inplace=True)
details


# In[20]:


# use the decode_run_length function imported from TestGrid_EDA notebook
details["values"] = details["statuses"].apply(decode_run_length)


# In[21]:


details["statuses"][0]


# In[22]:


details.alert[0]


# In[23]:


details.Type.unique()


# Sippy provides [dashboards](https://sippy.ci.openshift.org/?release=4.6) for openshift CI test/job data. This dashboard contains the health summary for CI's latest release. 
# 
# The various categories are as follow :
# * Job Pass Rates By Variant     
# * Curated TRT Tests     
# * Top Failing Tests Without a Bug 
# * Top Failing Tests With a Bug 
# * Job Pass Rates By Job Name 
# * Job Pass Rates By Most Reduced Pass Rate 
# * Infrequent Job Pass Rates By Job Name 
# * Canary Test Failures 
# * Job Runs With Failure Groups 
# * Test Impacting Bugs 
# * Test Impacting Components 
# * Job Impacting BZ Components

# ### Analyzing [Sippy](https://github.com/openshift/sippy) Code to reproduce their approach

# Test names can be deciphered using this [bit](https://github.com/openshift/sippy/blob/811be6ff0d094fb1bd172b5d68775d0f14464c90/pkg/testgridanalysis/testgridanalysisapi/types.go#L62) performed by sippy team.

# #### By Variants

# In[24]:


all_openshift_variants = [
    "aws",
    "azure",
    "fips",
    "gcp",
    "metal-assisted",
    "metal-upi",
    "metal-ipi",
    "never-stable",
    "openstack",
    "osd",
    "ovirt",
    "ovn",
    "ppc64le",
    "promote",
    "proxy",
    "realtime",
    "s390x",
    "serial",
    "upgrade",
    "vsphere-ipi",
    "vsphere-upi",
]
details = details.fillna("NA")


# In[25]:


df = details.copy()
df["variant_type"] = details["original-name"].apply(
    lambda x: [m for m in all_openshift_variants if m in x]
)
df["variant_type"] = df["variant_type"].map(str)
df["variant_type"]


# In[26]:


df["variant_type"].value_counts().plot()
plt.xticks(rotation=80)
plt.show()


# In the above cell, we can see that the variant : aws and upgrade have been marked in the column variant_type.
# This is how sippy looks up for some keywords that comprise of these variants and further categorize the tests accordingly.

# #### Curated TRT Tests

# In[27]:


## curated test substrings
curated_test = [
    "[Feature:SCC][Early] should not have pod creation failures during install",
    "infrastructure should work",
    "install should work",
    "Kubernetes APIs remain available",
    "OAuth APIs remain available",
    "OpenShift APIs remain available",
    "Pod Container Status should never report success for a pending container",
    "pods should never transition back to pending",
    "pods should successfully create sandboxes",
    "upgrade should work",
    "Cluster completes upgrade",
]


# In[28]:


df["Curated TRT"] = details["original-name"].apply(
    lambda x: [m for m in curated_test if m in x]
)
df["Curated TRT"] = df["Curated TRT"].map(str)


# In[29]:


df["Curated TRT"].head()


# In[30]:


df["Curated TRT"].unique()


# In[31]:


df["Curated TRT"].value_counts().plot()
plt.xticks(rotation=80)
plt.show()


# In[32]:


## Custom Job Setup Containers
custom_job = [
    "e2e-aws-upgrade-ipi-install-install-stableinitial",
    "e2e-aws-upgrade-rollback-ipi-install-install-stableinitial",
    "e2e-aws-proxy-ipi-install-install",
    "e2e-aws-workers-rhel7-ipi-install-install",
    "e2e-azure-upgrade-ipi-conf-azure",
    "e2e-gcp-upgrade-ipi-install-install-stableinitial",
    "e2e-metal-ipi-baremetalds-devscripts-setup",
    "e2e-metal-ipi-ovn-ipv6-baremetalds-devscripts-setup",
    "e2e-metal-ipi-ovn-dualstack-baremetalds-devscripts-setup",
    "e2e-vsphere-ipi-install-vsphere",
    "e2e-vsphere-upi-upi-install-vsphere",
    "e2e-vsphere-upi-serial-upi-install-vsphere",
    "e2e-vsphere-serial-ipi-install-vsphere",
]


# In[33]:


df["customJob"] = details["original-name"].apply(
    lambda x: [m for m in custom_job if m in x]
)
df["customJob"] = df["customJob"].map(str)
df.head()


# In[34]:


# Valid Buzilla Components
ValidBugzillaComponents = [
    "apiserver-auth",
    "assisted-installer",
    "Bare Metal Hardware Provisioning",
    "Build",
    "Cloud Compute",
    "Cloud Credential Operator",
    "Cluster Loader",
    "Cluster Version Operator",
    "CNF Variant Validation",
    "Compliance Operator",
    "config-operator",
    "Console Kubevirt Plugin",
    "Console Metal3 Plugin",
    "Console Storage Plugin",
    "Containers",
    "crc",
    "Dev Console",
    "DNS",
    "Documentation",
    "Etcd",
    "Federation",
    "File Integrity Operator",
    "Fuse",
    "Hawkular",
    "ibm-roks-toolkit",
    "Image",
    "Image Registry",
    "Insights Operator",
    "Installer",
    "ISV Operators",
    "Jenkins",
    "kata-containers",
    "kube-apiserver",
    "kube-controller-manager",
    "kube-scheduler",
    "kube-storage-version-migrator",
    "Logging",
    "Machine Config Operator",
    "Management Console",
    "Metering Operator",
    "Migration Tooling",
    "Monitoring",
    "Multi-Arch",
    "Multi-cluster-management",
    "Networking",
    "Node",
    "Node Feature Discovery Operator",
    "Node Tuning Operator",
    "oauth-apiserver",
    "oauth-proxy",
    "oc",
    "OLM",
    "openshift-apiserver",
    "openshift-controller-manager",
    "Operator SDK",
    "Performance Addon Operator",
    "Reference Architecture",
    "Registry Console",
    "Release",
    "RHCOS",
    "RHMI Monitoring",
    "Routing",
    "Samples",
    "Security",
    "Service Broker",
    "Service Catalog",
    "service-ca",
    "Special Resources Operator",
    "Storage",
    "Templates",
    "Test Infrastructure",
    "Unknown",
    "Windows Containers",
]


# In[35]:


df["Valid Bugzilla Components"] = details["original-name"].apply(
    lambda x: [m for m in ValidBugzillaComponents if m in x]
)
df["Valid Bugzilla Components"] = df["Valid Bugzilla Components"].map(str)


# In[36]:


df


# In[37]:


df["Valid Bugzilla Components"].unique()


# In[38]:


df["Valid Bugzilla Components"].value_counts().plot()
plt.xticks(rotation=80)
plt.ylim(0, 10)
plt.show()


# In this notebook, we have seen a few categories like Curated Jobs, classifying the jobs based on Variants and Bugzilla Components that can be seen on Sippy Dashboard. We can also try out more dashboards to see better results. This notebook takes in account job: https://testgrid.k8s.io/redhat-openshift-ocp-release-4.6-informing/table?&show-stale-tests=&tab=release-openshift-origin-installer-e2e-aws-upgrade-rollback-4.5-to-4.6 case. But we can change this link in cell[10] and see various different dashboards and perform this analysis sippy uses.
