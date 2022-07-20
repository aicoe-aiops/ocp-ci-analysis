#!/usr/bin/env python
# coding: utf-8

# # Prow Logs and GCS Data

# ### What do we have access to as data scientists when digging into the build artifacts?
# 
# In this notebook we will demonstrate how to discover and interact with the data (logs) made availble on [GCS/origin-ci-test](https://gcsweb-ci.apps.ci.l2s4.p1.openshiftapps.com/gcs/origin-ci-test/logs/) as well as provide some simple EDA to help folks get started analyzing this data.
# 
# This notebook is divided into 2 sections:
# 
# 1. Compare the different log files present throughout the archives and quantify how complete and comparable our log dataset is from build to build.
# 1. Download a sample dataset of the events and build logs to perform some lite EDA.
# 
# _Note: We will be collecting data from the "origin-ci-test" Bucket on Google Cloud Storage. But, after some out-of-notebook exploration it has become aparent that this is a massive amount of data that contains more than just the OpenShift CI logs we are intrested in here and programatically investigating that Bucket is not advised. Therefore, we recommend using the [web ui](https://gcsweb-ci.apps.ci.l2s4.p1.openshiftapps.com/gcs/origin-ci-test/logs/) to inspect what jobs are exposed and identify what is of interest to your analysis before collecting data via the google cloud stporage api. Here we will rely on web-scraping the UI to explore what's available to us based on what jobs are displayed on [TestGrid](https://testgrid.k8s.io/redhat-assisted-installer)._     

# ## Compare availability of log files across a build

# In[1]:


import json
import requests
import pandas as pd
from bs4 import BeautifulSoup
from google.cloud import storage
from pathlib import Path
from IPython.display import Markdown


# ### Example to access a single set of Prow artifacts
# 
# Let's make sure we understand how this works, and focus on a single job first.

# In[2]:


tab = '"redhat-openshift-ocp-release-4.6-informing"'
job = "periodic-ci-openshift-release-master-ci-4.6-upgrade-from-stable-4.5-e2e-gcp-upgrade"


# In[3]:


response = requests.get(
    f"https://gcsweb-ci.apps.ci.l2s4.p1.openshiftapps.com/gcs/origin-ci-test/logs/{job}"
)
soup = BeautifulSoup(response.text, "html.parser")
list_of_builds = [x.get_text()[1:-1] for x in soup.find_all("a")][1:-1]


# In[4]:


response = requests.get(
    f"https://gcsweb-ci.apps.ci.l2s4.p1.openshiftapps.com/gcs/origin-ci-test/logs/{job}/{list_of_builds[1]}"
)
response.url


# In[5]:


soup = BeautifulSoup(response.text, "html.parser")


# In[6]:


[x.get_text() for x in soup.find_all("a")]


# Great, we can now programmatically access the archives. Now, lets walk through all of the build archives for a single job and create a list of what they have on the first level of their directories.  

# In[7]:


build_data = {}

for build in list_of_builds:
    response = requests.get(
        f"https://gcsweb-ci.apps.ci.l2s4.p1.openshiftapps.com/gcs/origin-ci-test/logs/{job}/{build}"
    )
    soup = BeautifulSoup(response.text, "html.parser")
    artifacts = [x.get_text() for x in soup.find_all("a")]
    build_data[build] = artifacts


# In[8]:


builds_info = pd.Series({k: len(v) for (k, v) in build_data.items()})


# In[9]:


builds_info.value_counts()


# In[10]:


pd.Series(build_data).apply(" ".join).value_counts()


# In[11]:


cent1 = builds_info.value_counts() / len(builds_info)
cent1 = cent1.to_frame().reset_index()
cent1


# In[12]:


Markdown(
    "{}% of our records for this job appear to be complete and include the 'artifacts/' subdirectory,\
 lets dig in and see what they contain.".format(
        round(cent1.iloc[0, 1] * 100, 2)
    )
)


# In[13]:


build_data = {}

for build in list_of_builds:
    response = requests.get(
        f"https://gcsweb-ci.apps.ci.l2s4.p1.openshiftapps.com/gcs/origin-ci-test/logs/{job}/{build}/artifacts"
    )
    soup = BeautifulSoup(response.text, "html.parser")
    artifacts = [x.get_text() for x in soup.find_all("a")]
    build_data[build] = artifacts


# In[14]:


artifact_info = pd.Series({k: len(v) for (k, v) in build_data.items()})
artifact_info.value_counts()


# In[15]:


cent2 = artifact_info.value_counts() / len(artifact_info)
cent2 = cent2.to_frame().reset_index()
cent2


# In[16]:


Markdown(
    "The above shows us that there are about {}% of the artifacts dirs "
    "that have {} items and {}% that have {} items and so on "
    "(but it does not account for different combinations).".format(
        round(cent2.iloc[0, 1] * 100, 2),
        cent2.iloc[0, 0],
        round(cent2.iloc[1, 1] * 100, 2),
        cent2.iloc[1, 0],
    )
)


# In[17]:


pd.Series(build_data).apply(" ".join).value_counts(normalize=True)


# We can see from the results above that once we get down into the artifacts there is a far less uniformity to the data available to us for analysis. And this is all within a single job! Moving forward we will assume that this issue gets worse when comparing available artifacts across jobs and can dedicate a later notebook to proving out that assumption.  
# 
# This heterogeneity of objects available for each build will make it somewhat difficult to use these sets of documents as a whole to compare different CI behaviour. At this point, it makes sense to consider looking only at the same document (log) across job where available. 
# 

# ## Collect Data
# 
# ### Build logs
# 
# In the next section we are going to walkthrough accessing the `build-logs.txt` and the `events.json` as they appear to be nearly universally available. We will both download a small testing dataset as well show how to work directly with the data in memory.
# 
# Now that we know what logs we want to collect its simpler to use the google cloud storage api to access or data. 

# In[18]:


def connect_storage(bucket_name):
    storage_client = storage.Client.create_anonymous_client()
    bucket = storage_client.bucket(bucket_name)
    return {"bucket": bucket, "storage_client": storage_client}


def download_public_file(client, source_blob_name):
    """Downloads a public blob from the bucket."""
    blob = client["bucket"].blob(source_blob_name)
    if blob.exists(client["storage_client"]):
        text = blob.download_as_text()
    else:
        text = ""
    return text


# In[19]:


bucket_connection = connect_storage("origin-ci-test")


# In[20]:


# Read data into memory
build_log_data = {}
for build in list_of_builds:
    file = download_public_file(bucket_connection, f"logs/{job}/{build}/build-log.txt")
    build_log_data[build] = file


# In[21]:


build_log_data[list(build_log_data.keys())[0]]


# In[22]:


def get_counts(x):
    """
    Gets counts for chars, words, lines for a log.
    """
    if x:
        chars = len(x)
        words = len(x.split())
        lines = x.count("\n") + 1
        return chars, words, lines
    else:
        return 0, 0, 0


# In[23]:


## Create a dataframe with char, words, and lines
## count for the logs
data = []
for key, value in build_log_data.items():
    chars, words, lines = get_counts(value)
    data.append([key, chars, words, lines])

df = pd.DataFrame(data=data, columns=["build_log_id", "chars", "words", "lines"])
df


# #### See the stats for chars, words, lines

# In[24]:


df["chars"].describe()


# In[25]:


df["words"].describe()


# In[26]:


describe = df["lines"].describe()
describe = describe.to_frame()
describe


# In[27]:


Markdown(
    "From the initial analysis above, we see that we have log files "
    " with {} lines to {} lines with a mean of ~{} lines. "
    "This suggest high variability. The next thing we could look "
    " at would be the similarity between log files, "
    "performing analysis, templating, and clustering. "
    " We will address those questions in an upcoming notebook.".format(
        round(describe.iloc[3, 0]),
        round(describe.iloc[7, 0]),
        round(describe.iloc[1, 0]),
    )
)


# ### Events

# In[28]:


build_events_data = {}
for build in list_of_builds:
    file = download_public_file(
        bucket_connection, f"logs/{job}/{build}/artifacts/build-resources/events.json"
    )
    if file:
        build_events_data[build] = json.loads(file)
    else:
        build_events_data[build] = ""


# In[29]:


## Percentage of builds that have the events.json file
count = 0
for key, value in build_events_data.items():
    if value:
        count += 1
percent = count * 100 / len(build_events_data)
print(percent)


# In[30]:


# Analyzing the messages of a single build
single_build = sorted(build_events_data.keys())[0]
messages = [
    (i["metadata"]["uid"], i["message"])
    for i in build_events_data[single_build]["items"]
]
messages_df = pd.DataFrame(messages, columns=["UID", "message"])
messages_df


# In[31]:


messages_df["message"].describe()


# In[32]:


messages_df["message"].value_counts().reset_index()


# In[33]:


Markdown(
    "In the build data, we saw that about {}% builds have the events.json file."
    " We further analyzed all the events that happened for a particular build"
    " and found the frequencies of the messages. We can repeat the process for"
    " all the other builds and find most common messages and perform further"
    " analysis.".format(round(percent, 2))
)


# # Save sample data

# In[34]:


path = "../../../data/raw/gcs/build-logs/"
filename = "sample-build-logs.parquet"
dataset_base_path = Path(path)
dataset_base_path.mkdir(parents=True, exist_ok=True)
build_logs = pd.DataFrame.from_dict(build_log_data, orient="index", columns=["log"])
build_logs.to_parquet(f"{path}/{filename}")


# In[35]:


path = "../../../data/raw/gcs/events/"
filename = "sample-events.json"
dataset_base_path = Path(path)
dataset_base_path.mkdir(parents=True, exist_ok=True)

with open(f"{path}/{filename}", "w") as file:
    json.dump(build_events_data, file)


# ## Conclusion
# 
# In this notebook, we demonstrated how to programmatically access the gcs openshift origins ci archives, pull specific logs types into our notebook for analysis and save them for later use. 
# 
