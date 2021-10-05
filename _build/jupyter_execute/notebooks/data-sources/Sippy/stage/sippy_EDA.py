#!/usr/bin/env python
# coding: utf-8

# # Sippy Export of OpenShift Data - EDA
# 
# 
# In this notebook we will take a look at some of the openshift CI data distilled by the Sippy project with the following **goals** in mind. 
# 
# 1. Uncover the structure and contents of the dataset
# 2. Present some basic visualizations and  statistics on the provided data
# 3. Identify potential (or missing) features for further analysis or ML work
# 4. Capture questions/ points of clarification for the sippy team on the data. 
# 
# 
# In this notebook we will review a small sample dataset (sippydata.json) from August 10th, 2020 that has the following structure.   
# 
# 
# * **All**:  map of all the tests that were run in any job, single key is "all", value is a struct of pass count, fail count, pass percentage for tests run in "All jobs", plus an array of test results sorted by pass rate, test result struct includes the name of the test, its pass count, fail count, pass percentage. 
# 
# * **ByPlatform**: same data as above, except the map has a key for each platform we support (e.g. aws, gcp, etc), and the test results are only for jobs associated w/ those platforms.
# 
# * **ByJob**: same but sliced by the specific test job name.
# 
# * **BySig**: same but sliced by the "special interest group"(team) that is responsible for the tests
# 
# * **FailureGroups**:  job runs that had a large number of failing tests(usually indicates a fundamental problem in the product that impacted many tests), with metadata about the job run.
# 
# * **JobPassRate**: data for each job(not job run) sorted by the rate at which runs for that job as a whole passed(had no failing tests)
# 
# * **TopFaliingTestsWithBug**: tests which fail the most often (sorted by percentage, not raw count) across all job runs, and for which we have a known BZ
# 
# * **TopFailingTestsWithoutBug**: same as above, but for which we have no known BZ open
# 

# In[1]:


import pandas as pd
from pandas_profiling import ProfileReport


import seaborn as sns
import matplotlib.pyplot as plt

sns.set()


# In[2]:


d = pd.read_json("../../../../data/external/sippydata.json", orient="index")
d.columns = ["data"]
d


# ### Time of data sample

# In[3]:


d.loc["timestamp"].values[0]


# ### All
# 
# map of all the tests that were run in any job, single key is "all", value is a struct of pass count, fail count, pass percentage for tests run in "All jobs", plus an array of test results sorted by pass rate, test result struct includes the name of the test, its pass count, fail count, pass percentage. 

# In[4]:


# unroll the all data

_all = pd.DataFrame(d.loc["all"][0])
_all.head()


# In[6]:


sns.set(rc={"figure.figsize": (7, 7)})
plt.bar(
    ["failures", "successes"],
    [_all.loc["failures"].values[0], _all.loc["successes"].values[0]],
)
plt.title("Failures vs Successes: ALL", fontsize=20)
plt.ylabel("Total Count", fontsize=15)
plt.show()


# In[7]:


all_results = pd.DataFrame(_all.loc["results"][0])
all_results


# In[8]:


# Are all the names unique?
len(all_results["name"].unique()) == all_results["name"].shape[0]


# In[9]:


# Does BugList have any information?
all_results["BugList"].value_counts()


# In[10]:


# Does searchLink have any information?
all_results["searchLink"].value_counts()


# We can see that **'searchLink'** and **'BugList'** don't actually have any info and **'name'** is all unique values, so we will remove them from our exploratory analysis.  

# In[11]:


profile = ProfileReport(
    all_results.drop(["name", "BugList", "searchLink"], axis=1),
    title="All results",
)


# In[12]:


profile


#  We cab see from this report that ? ? ?

# # ByPlatform
# 
# Same data as above, except the map has a key for each platform we support (e.g. aws, gcp, etc), and the test results are only for jobs associated w/ those platforms.

# In[13]:


by_pltf = pd.DataFrame(d.loc["ByPlatform"][0])
by_pltf.head()


# In[14]:


barplot_s = by_pltf.loc[["successes"], :]
barplot_f = by_pltf.loc[["failures"], :]
barplot_p = by_pltf.loc[["testPassPercentage"], :]


# In[15]:


sns.set(rc={"figure.figsize": (10, 10)})
barplot_s.plot(kind="bar")
plt.ylabel("Total Count", fontsize=15)
plt.title("Successes: by platform", fontsize=20)
plt.xticks(rotation=45)
plt.show()


# In[16]:


sns.set(rc={"figure.figsize": (10, 10)})
barplot_f.plot(kind="bar")
plt.ylabel("Total Count", fontsize=15)
plt.title("Failures: by platform", fontsize=20)
plt.xticks(rotation=45)
plt.show()


# In[17]:


by_pltf.loc["testPassPercentage"].sort_values(ascending=False).apply(
    lambda x: x / 100
).plot(kind="bar")
sns.set(rc={"figure.figsize": (10, 10)})
plt.ylabel("Pass Percent", fontsize=15)
plt.title("Test Pass Percentage: by platform", fontsize=20)
plt.xticks(rotation=45)
plt.show()


# Now lets dig into the results by platform

# In[18]:


by_pltf.loc["results"]


# In[19]:


for p in by_pltf.loc["results"].index:
    platform = pd.DataFrame(by_pltf.loc["results"][p])
    platform = platform[platform["name"] != "Overall"]
    top_pass_percentage = (
        platform.sort_values(by="passPercentage")
        .head(10)
        .drop(["BugList", "searchLink"], axis=1)
    )
    to_plot = (
        top_pass_percentage.set_index("name")
        .drop(["successes", "failures", "flakes"], axis=1)
        .apply(lambda x: x / 100)
    )
    to_plot.plot(kind="bar")
    plt.ylabel("Pass Percent", fontsize=15)
    plt.title(f"Test Pass Percentage: {p}", fontsize=20)
    plt.xticks(rotation=85)
    plt.show()
    display(top_pass_percentage)


# # ByJob
# same but sliced by the specific test job name.

# In[20]:


by_job = pd.DataFrame(d.loc["ByJob"][0])
by_job.head()


# In[21]:


barplot_s = by_job.loc[["successes"]].T.sort_values(by="successes")
barplot_f = by_job.loc[["failures"]].T.sort_values(by="failures")
barplot_p = by_job.loc[["testPassPercentage"]].T.sort_values(by="testPassPercentage")


# In[22]:


# sns.set(rc = {"figure.figsize":(10,10)})
barplot_s.tail(10).plot.barh()
plt.title(" Top 10 Successes: by job", fontsize=20)
plt.xticks(rotation=45)
plt.show()


# In[23]:


barplot_s.head(10).plot.barh()
plt.title(" Bottom 10 Successes: by job", fontsize=20)
plt.xticks(rotation=45)
plt.show()


# In[24]:


barplot_p.apply(lambda x: x / 100).head(10).plot.barh()
plt.title(" Bottom test pass percent: by job", fontsize=20)
plt.xticks(rotation=45)
plt.show()


# now lets look at the results
# 
# since there are about 50 different jobs here, lets focus on displaying results just for those that lowest 5 test passing percentage and also have results.
# 
# -- question for sippy team. Why do some not have any results? 

# In[25]:


by_job_t = by_job.T.sort_values(by="testPassPercentage")
by_job_t = by_job_t.dropna()
by_job = by_job_t.T


# In[26]:


c = by_job.columns[:5]
for i in c:
    job = pd.DataFrame(by_job[i].results).drop(["BugList", "searchLink"], axis=1)
    job = job[job["name"] != "Overall"]

    print(i)
    print(f"Job pass percentage is {int(by_job[i].testPassPercentage)}%")
    display(pd.DataFrame(job))
    print("\n")

    to_plot = (
        job.set_index("name")
        .drop(["successes", "failures", "flakes"], axis=1)
        .apply(lambda x: x / 100)
    )
    to_plot.plot(kind="bar")
    plt.ylabel("Pass Percent", fontsize=15)
    plt.title(f"{i}", fontsize=20)
    plt.xticks(rotation=85)
    plt.show()


# The connection to "results" is a little unclear here. let's take the chart above for example: it shows that the job "release-openshift-ocp-installer-e2e-aws-upi-4.6" has a test pass percentage of 71%, but the results show only a single test with a pass percentage of 0. Which leads me to believe that the test pass percentage for the job is not a function purely of the results shown here.
# 
# We should follow up with sippy team for clarification.

# # BySig 
# 
# same as above but sliced by the "special interest group"(team) that is responsible for the tests

# In[1]:




