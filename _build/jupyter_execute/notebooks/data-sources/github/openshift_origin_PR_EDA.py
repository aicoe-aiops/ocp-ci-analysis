#!/usr/bin/env python
# coding: utf-8

# # GitHub Data for OpenShift Origins
# 
# In this notebook we will quickly walk through how to access and start analyzing the PR data for the OpenShift origins repository. We also explore the various fields in the PR dataset and the features that we can derive from them in order to frame an [ML problem](https://github.com/aicoe-aiops/ocp-ci-analysis/issues/236) where we predict the time taken to merge a PR.  
# 
# The ultimate hope for this work is that we will be able to connect the Github data (code changes) directly with the Bugs, Telemetry, TestGrid and Prow datasets in order to develop a complete picture of the OpenShift CI process.   

# ## Data Collection
# 
# The Thoth team already developed a tool for collecting data and perfoming analysis on GitHub repositoreis, [MI-Scheduler](https://github.com/thoth-station/mi-scheduler). So, we will not be accessing the data directly from GitHub, but from ceph storage after requesting that the OpenShift/Origin repo be included in the MI-Scheduler tool (see [issue](https://github.com/thoth-station/mi-scheduler/issues/130) for details).    

# In[1]:


import os
import re
import gzip
import json
import boto3
import trino
import pathlib
import datetime
from tqdm import tqdm
from io import BytesIO
from pathlib import Path
from functools import partial
from collections import Counter

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import STOPWORDS

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif, f_classif

from dotenv import load_dotenv, find_dotenv


# In[ ]:


class CephCommunication:
    """
    Class to establish communication with a ceph s3 bucket.
    It connects with the bucket and provides methods to read and write data in the parquet format.
    """

    def __init__(
        self, s3_endpoint_url, aws_access_key_id, aws_secret_access_key, s3_bucket
    ):
        self.s3_endpoint_url = s3_endpoint_url
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.s3_resource = boto3.resource(
            "s3",
            endpoint_url=self.s3_endpoint_url,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
        )
        self.bucket = s3_bucket
        ## Todo: Add try catch

    def upload_to_ceph(self, dataframe, s3_path, filename):
        """
        This helper function takes as input the data frame to be uploaded, and the output filename.
        It then saves the data frame in the defined ceph bucket.
        """
        parquet_buffer = BytesIO()
        dataframe.to_parquet(parquet_buffer)
        s3_obj = self.s3_resource.Object(self.bucket, f"{s3_path}/{filename}")
        status = s3_obj.put(Body=parquet_buffer.getvalue())
        return status

    def read_from_ceph(self, s3_path, filename):
        """
        Helper function to read from ceph and see if the saved data is correct.
        """
        buffer = BytesIO()
        s3_object = self.s3_resource.Object(self.bucket, f"{s3_path}/{filename}")
        s3_object.download_fileobj(buffer)
        df_temp = pd.read_parquet(buffer)
        return df_temp


def save_to_disk(dataframe, path, filename):
    """
    Helper function to save the dataframe
    as a parquet file to disk.
    """
    dataset_base_path = Path(path)
    dataset_base_path.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(f"{path}/{filename}")
    return True


# In[2]:


sns.set(rc={"figure.figsize": (20, 10)})
load_dotenv(find_dotenv())


# In[3]:


## CEPH Bucket variables
## Create a .env file on your local with the correct configs,
s3_endpoint_url = os.getenv("S3_ENDPOINT")
s3_access_key = os.getenv("S3_ACCESS_KEY")
s3_secret_key = os.getenv("S3_SECRET_KEY")
s3_bucket = os.getenv("S3_BUCKET")
s3_path = os.getenv("S3_PROJECT_KEY", "github")
s3_input_data_path = "raw_data"
REMOTE = os.getenv("REMOTE")
AUTOMATION = os.getenv("AUTOMATION")


# In[4]:


data_path = "../../../data/raw/GitHub/PullRequest.json.gz"
OUTPUT_DATA_PATH = "../../../data/processed/github"

if REMOTE:
    print("getting dataset from ceph")
    s3 = boto3.resource(
        "s3",
        endpoint_url=s3_endpoint_url,
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key,
    )
    content = s3.Object(s3_bucket, "thoth/mi/openshift/origin/PullRequest.json")
    file = content.get()["Body"].read().decode("utf-8")
    prs = json.loads(file)

    if not AUTOMATION:
        with gzip.open(data_path, "wb") as out_file:
            out_file.write(json.dumps(prs).encode("utf-8"))

else:
    print("getting dataset from local")
    with gzip.open(data_path, "r") as f:
        prs = json.loads(f.read().decode("utf-8"))

pr_df = pd.DataFrame(prs).T


# In[5]:


pr_df.shape


# In[6]:


pr_df.head(2)


# ## Inspect the dataset
# 
# Let's unroll a single row and see what kind of information we have access to. 

# In[7]:


pr = pr_df.iloc[4]
pr


# </br>
# 
# #### **Available Fields**
# 
# * **title**: text providing a short description of the of the PR.
# 
# * **body**: text providing additional details regarding the changes. 
# 
# * **size**: categorical label generated by GitHub summerizing the size of the changes made to the code base. 
# 
# * **created/closed/merged_at**: timestamps for when the PR was opened, merged and finally closed. 
# 
# * **created/closed_by**: usernames for contributors who created the PR and closed it. 
# 
# * **commits_number**: integer indicating the number of commits in the PR.
# 
# * **changed_files_number**: integer indicating the number of files changed.
# 
# The remaining fields appear to be collections of items, let's unroll each one and see what additional information they provide.
# 

# In[8]:


pr["interactions"]


# **interactions**: a dictionary with usernames for keys and the number of times they interacted with the PR as the value.

# In[9]:


pr["reviews"]


# **reviews**: a dictionary of reviews that includes an id number key along with fields for author, word_count, submitted timestamp and state of review.

# In[10]:


pr["labels"]


# **labels**: a list of labels tagged to the PR describing some of its attributes

# In[11]:


pr["commits"]


# **commits**: a list of commit hashes that point to specific changes made to the repo's history

# In[12]:


print(len(pr["changed_files"]))
pr["changed_files"][0:10]


# **changed_files**: a list of the paths and filenames for every file changed by this PR.

# We now know what we have access to in this dataset. It is a collection of numerical, categorical and textual features used to describe a PR. This gives us a lot of potential avenues to explore from an EDA and Data Science perspective. But is also creates an additional challenge insofar as we'll need to do a bit of feature engineering to get this data into format that is ready to be ingested by any ML models. 

# ## Feature engineering
# 
# In the next section we will do some basic feature engineering and correlation analyis on our dataset to see if anything interesting jumps out at us. 

# **Time to Merge**
# 
# We will derive a new feature called `time_to_merge` which will be the difference between the `created_at` and `merged_at` fields. 

# In[13]:


# derive time_to_merege
pr_df["time_to_merge"] = pr_df["merged_at"] - pr_df["created_at"]
pr_df["time_to_merge"] = pr_df["time_to_merge"].astype("float")
# NOTE: we cant fill NA with 0 because that would imply the PR got merged instantaneously


# Let's look at the distribution of the time to merge column. This can help us determine what the setup for ML problems such as [#236](https://github.com/aicoe-aiops/ocp-ci-analysis/issues/236) could be like. That is, should the time to merge problem be framed as a regression or classification problem.
# 
# First, lets try to determine what the granularity of the output values should be, in case of a regression setup. That is, should the model predict time to merge in number of seconds, number of hours, number of days, or some other chunk of time.

# In[14]:


# what does the distribution of values look like if we choose days as the granularity
ttm_days = np.ceil(pr_df.time_to_merge / (60 * 60 * 24))

# value counts - to how many PRs have value 1, 2, 3, etc. for days to merge
vc = ttm_days.value_counts()

# normalized value counts (i.e. percent of PRs instead of raw number)
nvc = vc / vc.sum()
nvc


# In[15]:


# only plot top 100 values so that graph is legible
nvc.sort_index().iloc[:100].plot(kind="bar")
plt.xlabel("Days")
plt.ylabel("Number of PRs")
plt.title("Number of PRs per Day")
plt.show()


# From the above graph we can see that if we bucket the data into days, then roughly 40% of the data points would have the value "1", 15% would have the value "2" and so on. Therefore we need to select a finer granularity. 

# In[16]:


# what if we chose some intermediate value e.g. 3 hours as the granularity?
ttm_3hours = np.ceil(pr_df.time_to_merge / (60 * 60 * 3))

# value counts - to how many PRs have value 1, 2, 3, etc. for num. 3-hours to merge
vc = ttm_3hours.value_counts()

# normalized value counts (i.e. percent of PRs instead of raw number)
nvc = vc / vc.sum()
nvc


# In[17]:


# only plot top 100 values so that graph is legible
nvc.sort_index().iloc[:100].plot(kind="bar")
plt.xlabel("3 Hour Intervals")
plt.ylabel("Number of PRs")
plt.title("Number of PRs per 3 Hour Interval")
plt.show()


# From the above outputs we can see that if we bucket the data into 3 hours, then roughly 10% of the data points would have the value "1", 10% would have the value "2", 5% would have the value 3, and so on. This is a slightly better value distribution than before, but could be made even better. Lets explore what this would look like with an "hours" granularity.

# In[18]:


# what if we chose hours as the granularity?
ttm_hours = np.ceil(pr_df.time_to_merge / (60 * 60))

# value counts - to how many PRs have value 1, 2, 3, etc. for hours to merge
vc = ttm_hours.value_counts()

# normalized value counts (i.e. percent of PRs instead of raw number)
nvc = vc / vc.sum()
nvc


# In[19]:


# only plot top 100 values so that graph is legible
nvc.sort_index().iloc[:100].plot(kind="bar")
plt.xlabel("Hours")
plt.ylabel("Number of PRs")
plt.title("Number of PRs per Hour")
plt.show()


# From the above outputs we can see that if we bucket the data into hours, then roughly 3% of the data points would have the value "2", 3% would have the value "3", 3% would have the value "4", and so on. This seems like a reasonable distribution of values for a regression problem, since the data does not get skewed towards any one particular value.
# 
# As we make it more granular (e.g. minutes), the distribution would get even more uniform. However, a granularity finer than 1 hour might not provide any additional value to the end user. So we will go with granularity of 1 hour for our regression setup.

# Now, lets try to determine, if we were to set this up as a classification problem, what the output classes should be. In the following cell, we'll try to split the time-to-merge values into 10 equally populated buckets.

# In[20]:


# lets look at what the 10th, 20th, 30th etc percentiles are, to see what the bin boundaries (in terms of hours)
# would be if we were to split the data in 10 equally populated bins
n_buckets = 10

quantiles = pr_df.time_to_merge.quantile(q=np.arange(0, 1 + 1e-100, 1 / n_buckets))
quantiles / 3600


# In[21]:


# gaps between the buckets (i.e. size of each bucket) in hours
(quantiles - quantiles.shift()) / 3600


# If we want to frame our ML problem as a classification problem, we can use the above buckets to define the classes. That is, class 0 is "merged within 3 hours", class 1 is "merged within 3-7 hours", etc.

# Based on the EDA so far, we cannot conclusively reject either classification or regression. We can explore both of these approaches in depth in forthcoming notebooks and re-evaluate the best setup for this problem.

# **Body**
# 
# Here, we will derive a feature called `body_size` which will simply be the number of words in the body of the PR. We can use this numerical value to describe the body of the PR.

# In[22]:


# derive body_size
pr_df["body_size"] = pr_df["body"].fillna("").apply(lambda x: len(x.split()))


# In[23]:


# derive body_size
pr_df["body_size"] = pr_df["body"].fillna("").apply(lambda x: len(x.split()))


# In[24]:


pr_df.head(2)


# **Size**
# 
# We can attempt to convert the categorical size label generated by GitHub, which summerizes the size of the changes made to the code base, to a numerical value. 

# In[25]:


convert_size = {"size": {"XS": 0, "S": 1, "M": 2, "L": 3, "XL": 4, "XXL": 5}}
pr_df = pr_df.replace(convert_size)


# **Title**
# 
# Lets try to see if the PR title text can be used to extract any useful information regarding the PR.

# In[26]:


# first lets preprocess the text available in the titles

# convert to lowercase
preproc_titles = pr_df["title"].str.lower()

# remove punctuations and symbols like : ; , # ( ) [ ] etc
preproc_titles = preproc_titles.str.replace(r'[`#-.?!,:;\/()\[\]"\']', " ", regex=True)

# remove hash-like strings i.e. 25+ char long strings containing 0-9 or a-f
preproc_titles = preproc_titles.str.replace(r"[0-9a-f]{25,}", " ", regex=True)

preproc_titles


# In[27]:


# what are the most commonly appearing words in the titles

# combin all titles and split into words
words = preproc_titles.str.cat(sep=" ").split()

# remove stopwords and numbers (e.g. bugzilla ids)
words = [w for w in words if w not in set(STOPWORDS) and not w.isnumeric()]

# word frequencies
unique_words, counts = np.unique(words, return_counts=True)
vc = pd.Series(counts, index=unique_words).sort_values(ascending=False)

# plot the 20 most common words
sns.barplot(
    x=vc.index[:20],
    y=vc.values[:20],
)
plt.title("Top word frequencies in the PR Titles")
plt.xlabel("words")
plt.ylabel("count")
plt.show()


# From the above graph, we can see that titles often contain some keywords that can hint towards the type of changes being made in the PR. We will create a feature which is a vector indicating how many times each word appeared in the title. This way, even if a less prominent word is more correlated with time_to_merge, we will be able to capture that relationship.

# In[28]:


# add word count columns
for word in tqdm(unique_words):
    pr_df[f"title_wordcount_{word}"] = preproc_titles.apply(
        lambda x: x.split().count(word)
    )


# In[29]:


# collapse the high dim vector into one column
wordcount_columns = [c for c in pr_df.columns if "wordcount" in c]
pr_df["title_word_counts_vec"] = pr_df[wordcount_columns].apply(
    lambda x: x.tolist(), axis=1
)

# drop the individual wordcount columns
pr_df = pr_df.drop(columns=wordcount_columns)


# In[30]:


pr_df.head(2)


# **Created By**
# 
# Is the time taken to merge a PR affected by the author of the PR? E.g. if a code reviewer or someone who has already made several contributions to a repo creates a PR, is it likely to get merged relatively faster?

# In[31]:


# approvers and reviewers list from the repo
approvers = [
    "bparees",
    "deads2k",
    "derekwaynecarr",
    "eparis",
    "knobunc",  # Network, Multi-cluster, Storage
    "mfojtik",
    "pweil-",
    "sjenning",
    "soltysh",
    "sttts",
    "smarterclayton",
]

reviewers = [
    "bparees",
    "deads2k",
    "sjenning",
    "smarterclayton",
    "soltysh",
]


# In[32]:


# is the pr creator on the approvers or reviewers list
pr_df["is_reviewer"] = pr_df["created_by"].isin(reviewers)
pr_df["is_approver"] = pr_df["created_by"].isin(approvers)


# In[33]:


pr_df.head(2)


# We also want to derive a feature called `num_prev_merged_prs` captures the number of PRs that the creator created before this PR.

# In[34]:


# how many prs did the creator create before this pr
def get_prev_pr_info(creator_pr_df):
    num_prev_merged_prs = (
        creator_pr_df["time_to_merge"].expanding().count().shift(1).fillna(0)
    )
    return num_prev_merged_prs


# need to sort before applying groupby
# because merging is not possible as df index types are different
pr_df = pr_df.sort_values(by="created_at")

num_prev_merged_prs = pr_df.groupby("created_by").apply(get_prev_pr_info)
num_prev_merged_prs = num_prev_merged_prs.reset_index(drop=True).rename("num_prev_prs")

pr_df["num_prev_merged_prs"] = num_prev_merged_prs.values


# In[35]:


pr_df.head(2)


# **Created At**
# 
# This column can give us information on when the PR was created. Using the `created_at` field, we can derive other features like `time_of_day`, `day_of_week`, `day_of_month`, `month_in_year` each of which can be categorical features that can be useful to our analysis.

# In[36]:


# convert to datetime object
pr_df["created_at_datetime"] = (
    pr_df["created_at"].fillna("").apply(lambda x: datetime.datetime.fromtimestamp(x))
)
# get day as categorical variable
pr_df["created_at_day"] = pr_df["created_at_datetime"].apply(lambda x: x.day)

# get month as categorical variable
pr_df["created_at_month"] = pr_df["created_at_datetime"].apply(lambda x: x.month)

# get weekday as categorical variable
pr_df["created_at_weekday"] = pr_df["created_at_datetime"].apply(lambda x: x.weekday())

# get hour of day as categorical variable
pr_df["created_at_hour"] = pr_df["created_at_datetime"].apply(lambda x: x.hour)


# In[37]:


pr_df.head(2)


# **Commits Number**  
# Commits Number gives the number of commits created within the PR. This could be a numerical feature for our model.

# In[38]:


pr_df.commits_number.value_counts()


# In[39]:


pr_df.commits_number.value_counts().sort_index(ascending=True).plot(kind="bar")
plt.title("Commit frequency across PRs")
plt.xlabel("No. of commits")
plt.ylabel("No. of PRs")
plt.show()


# As we see above, in this dataset, there is not much variability in the commits number. Majority of the PRs (77%) have 1 commit, 11% have 2 commits and rest(12%) have more than 2 commits. We can break commits number down into 3 categorical variables, `contains_1_commit`, `contains_2_commits`, `contains_3_or_more_commits`. However, there might be some value in knowing whether a PR had 5 commits vs. 10 commits. So we will treat it as a numerical variable to begin with.

# In[40]:


pr_df.head(2)


# **Changed Files Number**  
# Changed files is a numberical variable which captures the numer of files that were modified in the PR.

# In[41]:


# convert changed_files_number to int
pr_df["changed_files_number"] = pr_df["changed_files_number"].astype("int")

pr_df.changed_files_number.value_counts()


# In[42]:


pr_df.changed_files_number.value_counts().iloc[:50].sort_index(ascending=True).plot(
    kind="bar"
)
plt.title("Frequency of changed files across PRs")
plt.xlabel("No. of changed files")
plt.ylabel("No. of PRs")
plt.show()


# `changed_files_number` has more variability than the `commits_number`. We can incorporate this as a numerical feature.

# **Changed_files**
# 
# `changed_files` is a list of the paths and filenames for every file changed by this PR. From `changed_files`, we know, for each PR, which file types were changed which can be encoded as a frequency vector. Also we know which directory the change is made in, so that can be encoded as a categorical variable.

# In[43]:


pr_df.changed_files[7]


# In[44]:


def filetype(filepath):

    # if standard file extension, return file extension
    if pathlib.Path(filepath).suffix:
        return pathlib.Path(filepath).suffix

    else:
        # if single file at the root of repo
        if "/" not in filepath:
            # if a file like README/Dockerfile etc
            if "." not in filepath:
                return filepath
        # if file not at root
        else:
            # get file name
            file = filepath.rsplit("/", 1)[1]
            if "." not in file:
                return file
            elif "." in file:
                return "." + file.rsplit(".", 1)[1]


# In[45]:


# get list of all file types that were modified across all openshift origin PRs
changed_files = list(pr_df["changed_files"])


# In[46]:


all_changed_files = [filetype(filepath) for pr in changed_files for filepath in pr]


# In[47]:


count = Counter(all_changed_files)


# In[48]:


# save 100 most commonly occuring file extensions across PRs
top_fileextensions_freq = count.most_common(100)
top_fileextensions = [ext[0] for ext in top_fileextensions_freq]


# In[49]:


top_fileextensions


# In[50]:


def file_type_freq(list_of_filepaths):
    """
    For a given PR's list of changed files,
    return a changed files frequency vector
    ( a vector which holds the frequency of
    changed files of the type  for every
    file type in top_fileextensions )
    """

    file_extensions = [filetype(f) for f in list_of_filepaths]
    ext_dict = {key: 0 for key in top_fileextensions}
    for f in file_extensions:
        if f in ext_dict:
            ext_dict[f] += 1

    return list(ext_dict.values())


# In[51]:


pr_df["changed_file_type_vec"] = pr_df["changed_files"].apply(
    lambda x: file_type_freq(x)
)


# In[52]:


pr_df.head(2)


# In[53]:


# We will also create a categorical variable for a modification in
# each of the directories present in the openshift origins repo

directories = [
    ".github",
    "cmd",
    "docs",
    "e2echart",
    "examples",
    "hack",
    "images",
    "pkg",
    "test",
    "tools",
    "vendor",
    "root",
]


# In[54]:


def directory_from_filepath(list_of_filepaths):

    directories = []
    for filepath in list_of_filepaths:
        if "/" in filepath:
            directories.append(filepath.split("/", 1)[0])
        else:
            directories.append("root")

    return directories


# In[55]:


# does the PR make a change to the given directory
pr_df["change_in_.github"] = pr_df["changed_files"].apply(
    lambda x: 1 if ".github" in directory_from_filepath(x) else 0
)
pr_df["change_in_cmd"] = pr_df["changed_files"].apply(
    lambda x: 1 if "cmd" in directory_from_filepath(x) else 0
)
pr_df["change_in_docs"] = pr_df["changed_files"].apply(
    lambda x: 1 if "docs" in directory_from_filepath(x) else 0
)
pr_df["change_in_e2echart"] = pr_df["changed_files"].apply(
    lambda x: 1 if "e2echart" in directory_from_filepath(x) else 0
)
pr_df["change_in_examples"] = pr_df["changed_files"].apply(
    lambda x: 1 if "examples" in directory_from_filepath(x) else 0
)
pr_df["change_in_hack"] = pr_df["changed_files"].apply(
    lambda x: 1 if "hack" in directory_from_filepath(x) else 0
)
pr_df["change_in_images"] = pr_df["changed_files"].apply(
    lambda x: 1 if "images" in directory_from_filepath(x) else 0
)
pr_df["change_in_pkg"] = pr_df["changed_files"].apply(
    lambda x: 1 if "pkg" in directory_from_filepath(x) else 0
)
pr_df["change_in_test"] = pr_df["changed_files"].apply(
    lambda x: 1 if "test" in directory_from_filepath(x) else 0
)
pr_df["change_in_tools"] = pr_df["changed_files"].apply(
    lambda x: 1 if "tools" in directory_from_filepath(x) else 0
)
pr_df["change_in_vendor"] = pr_df["changed_files"].apply(
    lambda x: 1 if "vendor" in directory_from_filepath(x) else 0
)
pr_df["change_in_root"] = pr_df["changed_files"].apply(
    lambda x: 1 if "root" in directory_from_filepath(x) else 0
)


# In[56]:


pr_df.head(2)


# ## SQL-ify columns for creating SQL Tables from the dataframe

# In[57]:


pr_df.info()


# In[58]:


cols_to_drop = [
    "interactions",
    "reviews",
    "labels",
    "commits",
    "changed_files",
    "title_word_counts_vec",
    "changed_file_type_vec",
]
sql_df = pr_df.drop(columns=cols_to_drop)
sql_df.head()


# In[59]:


sql_df = sql_df.infer_objects()


# In[60]:


# convert to timestamps
ts_cols = ["created_at", "closed_at", "merged_at"]

for c in ts_cols:
    sql_df[c] = pd.to_datetime(sql_df[c], unit="s")


# In[61]:


sql_df.dtypes


# ## Save dataframe to S3 and create a Trino table

# In[62]:


## Methods to clean column names
# Author: Erik Erlandson <eje@redhat.com>

_rmpunc = re.compile(r"[,.()&$/+-]+")
_wsdedup = re.compile(r"\s+")
_usdedup = re.compile(r"__+")

_p2smap = {
    "object": "varchar",
    "int64": "bigint",
    "float64": "double",
    "datetime64[ns]": "timestamp",
    "bool": "boolean",
}


def pandas_type_to_sql(pt):
    st = _p2smap.get(pt)
    if st is not None:
        return st
    raise ValueError("unexpected pandas column type '{pt}'".format(pt=pt))


def generate_table_schema_pairs(df):
    ptypes = [str(e) for e in df.dtypes.to_list()]
    stypes = [pandas_type_to_sql(e) for e in ptypes]
    pz = list(zip(df.columns.to_list(), stypes))
    return ",\n".join(["    {n} {t}".format(n=e[0], t=e[1]) for e in pz])


def snakify(name, maxlen):
    w = name.casefold().rstrip().lstrip()
    w = w.replace("-", "_")
    w = _rmpunc.sub("", w)
    w = _wsdedup.sub("_", w)
    w = _usdedup.sub("_", w)
    w = w.replace("average", "avg")
    w = w.replace("maximum", "max")
    w = w.replace("minimum", "min")
    w = w.replace("absolute", "abs")
    w = w.replace("source", "src")
    w = w.replace("distribution", "dist")
    w = w[:maxlen]
    return w


def snakify_columns(df, inplace=False, maxlen=63):
    icols = df.columns.to_list()
    ocols = [snakify(e, maxlen=maxlen) for e in icols]
    if len(set(ocols)) < len(ocols):
        raise ValueError("remapped column names were not unique!")
    rename_map = dict(list(zip(icols, ocols)))
    return df.rename(columns=rename_map, inplace=inplace)


# In[63]:


# Create a Trino client
conn = trino.dbapi.connect(
    auth=trino.auth.BasicAuthentication(
        os.environ["TRINO_USER"], os.environ["TRINO_PASSWD"]
    ),
    host=os.environ["TRINO_HOST"],
    port=int(os.environ["TRINO_PORT"]),
    http_scheme="https",
    verify=True,
)
cur = conn.cursor()


# In[64]:


snakify_columns(sql_df, inplace=True)


# In[65]:


schema = generate_table_schema_pairs(sql_df)
print(schema)


# In[66]:


filename = "PR_DF.parquet"
key = "github/openshift-origin/pr_df"

cc = CephCommunication(s3_endpoint_url, s3_access_key, s3_secret_key, s3_bucket)
cc.upload_to_ceph(sql_df, key, filename)


# In[67]:


tabledef = """create table if not exists data_science_general_morty.default.pr_df(
{schema}
) with (
    format = 'parquet',
    external_location = 's3a://{s3_bucket}/github/openshift-origin/pr_df/'
)""".format(
    schema=schema,
    s3_bucket=os.environ["S3_BUCKET"],
)
cur.execute(tabledef)
cur.fetchall()


# In[68]:


## Check if table 1 is there
cur.execute("select * from data_science_general_morty.default.pr_df LIMIT 5")
cur.fetchall()[1]


# ## Infer dataframe columns as Features

# We can see above that some PRs have 0 for all directory fields since the directory structure of the repo has changed over time and we are interested in only the latest directory structure.

# In[69]:


# subset DataFrame to use as features
feature_df = pr_df[
    [
        "time_to_merge",
        "changed_files_number",
        "body_size",
        "size",
        "title_word_counts_vec",
        "is_reviewer",
        "is_approver",
        "num_prev_merged_prs",
        "created_at_day",
        "created_at_month",
        "created_at_weekday",
        "created_at_hour",
        "commits_number",
        "changed_file_type_vec",
        "change_in_.github",
        "change_in_cmd",
        "change_in_docs",
        "change_in_e2echart",
        "change_in_examples",
        "change_in_hack",
        "change_in_images",
        "change_in_pkg",
        "change_in_test",
        "change_in_tools",
        "change_in_vendor",
        "change_in_root",
    ]
]


# ### Add PR TTM Class Column 
# 
# Lets split up the time to merge values into 10 discrete bins (as discussed above) to use as our prediction classes.

# In[70]:


def get_ttm_class(ttm):
    """
    Assign a ttm "class" / "category" / "bin" to the input numerical ttm value
    E.g. if the time to merge was 1.5 hours, this function will return
    class "0" which represents "merged in 0-3 hours"
    """
    if ttm < 3:
        return 0
    elif ttm < 6:
        return 1
    elif ttm < 15:
        return 2
    elif ttm < 24:
        return 3
    elif ttm < 36:
        return 4
    elif ttm < 60:
        return 5
    elif ttm < 112:
        return 6
    elif ttm < 190:
        return 7
    elif ttm < 462:
        return 8
    else:
        return 9


feature_df["ttm_class"] = (feature_df["time_to_merge"] / 3600).apply(get_ttm_class)
feature_df.head()


# In[71]:


# visualize data points distribution across classes
feature_df["ttm_class"].hist()
plt.title("Distribution of PRs across TTM Labels")
plt.xlabel("Time to Merge Class")
plt.ylabel("Number of PRs")
plt.show()


# From the above graph, we can see that data is fairly uniformly distributed across all classes. At the same time, the class definitions are also easily interpretable by human experts. So this seems like a good setup for our classification task.

# Let's look at some pair-wise scatter plots of our data to see if there are any obvious correlations between our features. 

# In[72]:


fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
ax1.scatter(pr_df["time_to_merge"], pr_df["body_size"])
ax1.set(xlabel="time to merge (seconds)", ylabel="body_size")
ax2.scatter(pr_df["time_to_merge"], pr_df["changed_files_number"])
ax2.set(xlabel="time to merge (seconds)", ylabel="changed_file_number")
ax3.scatter(pr_df["body_size"], pr_df["changed_files_number"])
ax3.set(xlabel="body size", ylabel="changed_file_number")

ax4.scatter(pr_df["size"], pr_df["changed_files_number"])
ax4.set(xlabel="size", ylabel="changed_file_number")
ax5.scatter(pr_df["size"], pr_df["time_to_merge"])
ax5.set(xlabel="size", ylabel="time_to_merge (seconds)")
ax6.scatter(pr_df["size"], pr_df["body_size"])
ax6.set(xlabel="size", ylabel="body_size")

fig.suptitle("Pair-Wise Scatter Plots")
plt.show()


# Its tough to see any meaningful correlations here. Despite this "L" shaped scatter plot the vast majority of the points are centered around (0,0) for the 3 pairs of features in the top row. There is some positive correlation seen between size and changed_file_number and size and body_size since the points appear to increase in the same direction. However, there does not appear to be any correlation seen between time_to_merge and size since the time_to_merge does not change even as size increases. Let's calculate the correlation value for each pair of variables to see some actual numbers.

# In[73]:


feature_df.describe()


# In[74]:


feature_df.corr()


# From the correlation values, above we do not see any strong correlation between `time to merge` and other variables. We can see that body_size and size are _most_ correlated with time_to_merge, but 0.11 is a rather weak correlation value. This is good if we are going to use these as input features to a model, but not so good if we wanted to try and predict one value from the other two.  In the next section, we will calculate feature importance scores for our features to select the most important features for our model.

# In[75]:


# NOTE: rows where the `time_to_merge` value is nan represent the PR's that did not get merged
# since predicting time to close/reject is out of the scope of this exploration
# we will drop those rows from further analysus
feature_df = feature_df[~feature_df["time_to_merge"].isna()]
feature_df.shape


# ## Evaluating Feature Importances
# 
# ### Categorical Features
# 
# First, lets try to determine what categorical featuers are powerful influencers for time to merge of a PR

# In[76]:


# feature selection helper function
def select_features(x, y, func):
    fs = SelectKBest(score_func=func, k="all")
    fs.fit(x, y)
    x_fs = fs.transform(x)
    return x_fs, fs


selected_features = []


# In[77]:


cat_df = feature_df[
    [
        "size",
        "is_reviewer",
        "is_approver",
        "created_at_day",
        "created_at_month",
        "created_at_weekday",
        "created_at_hour",
        "change_in_.github",
        "change_in_cmd",
        "change_in_docs",
        "change_in_e2echart",
        "change_in_examples",
        "change_in_hack",
        "change_in_images",
        "change_in_pkg",
        "change_in_test",
        "change_in_tools",
        "change_in_vendor",
        "change_in_root",
        "ttm_class",
    ]
]

cat_df.head()


# In[78]:


# split into input (x) and output (y) variables
X_inp = cat_df.drop(columns=["ttm_class"])
y = cat_df["ttm_class"]


# In[79]:


# rank using mutual_info_classif
X_fs, fs = select_features(X_inp, y, mutual_info_classif)


# In[80]:


for i in range(len(fs.scores_)):
    print("Feature %s: %f" % (X_inp.columns[i], fs.scores_[i]))


# In[81]:


# visualize scores of all features
sns.barplot(x=X_inp.columns, y=fs.scores_)
plt.title("Feature importance scores for Mutual Information Statistic")
plt.xticks(rotation=60)
plt.xlabel("Features")
plt.ylabel("Scores")
plt.show()


# In[82]:


# Lets get a list of the top 10 feature indices based on Mutual Info Stat feature scores
top_mis = fs.scores_.argsort()[-10:][::-1]


# In[83]:


# rank using chi_sq
X_fs, fs = select_features(X_inp, y, chi2)


# In[84]:


for i in range(len(fs.scores_)):
    print("Feature %s: %f" % (X_inp.columns[i], fs.scores_[i]))


# In[85]:


# visualize scores of all features
sns.barplot(x=X_inp.columns, y=fs.scores_)
plt.title("Feature importance scores for Chi Squared")
plt.xticks(rotation=60)
plt.xlabel("Features")
plt.ylabel("Scores")
plt.show()


# In[86]:


# Lets get a list of the top 10 feature indices based on Chi Sq feature scores
top_chi2 = fs.scores_.argsort()[-10:][::-1]


# Let's get a Union of the top 10 features derived from features based on both scores of both Mutual Info Statistic and Chi Squared

# In[87]:


union = set(top_mis).union(set(top_chi2))
print(len(union))
union


# We now have the top 12 features based on their feature importance scores. We can eliminate the remaining 7 features from the categorical features which have low feature importance scores for both  Mutual Info Statistic and Chi Squared 

# In[88]:


cat_df.head()


# In[89]:


# update list of features to keep for training
selected_features += [
    "size",
    "is_reviewer",
    "is_approver",
    "created_at_day",
    "created_at_month",
    "created_at_weekday",
    "created_at_hour",
    "change_in_.github",
    "change_in_docs",
    "change_in_pkg",
    "change_in_test",
    "change_in_vendor",
    "change_in_root",
]


# ### Numerical Features
# 
# Now, lets try to determine what numerical featuers are powerful influencers for time to merge of a PR

# In[90]:


numerical_df = feature_df[
    [
        "changed_files_number",
        "body_size",
        "num_prev_merged_prs",
        "commits_number",
        "ttm_class",
    ]
]

numerical_df.head()


# In[91]:


# split into input features and target
X_inp = numerical_df.drop(columns=["ttm_class"])
y = numerical_df["ttm_class"]


# In[92]:


# create k best feature selector object based on mutual information
numerical_feats_mi, k_best_selector_mi = select_features(
    X_inp,
    y,
    partial(
        mutual_info_classif,
        discrete_features=False,
        random_state=42,
        n_neighbors=3,
    ),
)

# visualize the MI scores of all features
sns.barplot(x=X_inp.columns, y=k_best_selector_mi.scores_)
plt.title("Feature importance based on Mutual Information")
plt.xticks(rotation=60)
plt.xlabel("Features")
plt.ylabel("Scores")
plt.show()


# In[93]:


# create k best feature selector object based on anova f statistic
numerical_feats_f, k_best_selector_f = select_features(
    X_inp,
    y,
    f_classif,
)

# visualize the f stat scores and pvalues of all features
fig, axes = plt.subplots(2, 1, sharex=True)
fig.suptitle("Feature importance based on ANOVA F stat")
sns.barplot(x=X_inp.columns, y=k_best_selector_f.scores_, ax=axes[0])
sns.barplot(x=X_inp.columns, y=k_best_selector_f.pvalues_, ax=axes[1])
plt.xticks(rotation=60)
plt.xlabel("Features")
axes[0].set_ylabel("Scores")
axes[1].set_ylabel("p-values")
plt.show()


# From the above cells, we can see that all numerical featuers seem to have similar importances based on the mutual information criterion. Based on ANOVA F statistic, all but one (`body_size`) of them have similar scores. Since there are only 4 of these features, we will keep all of them in our features dataframe.

# In[94]:


# update list of features to keep for training
selected_features += [
    "changed_files_number",
    "body_size",
    "num_prev_merged_prs",
    "commits_number",
]


# ### Count Vector Type Features
# 
# Finally, lets determine the most important features from the count vector type columns.

# In[95]:


# explode the list column so that there is 1 column representing frequency of each file type
filetype_df = pd.DataFrame(feature_df.changed_file_type_vec.to_list())
filetype_df.index = feature_df.index
filetype_df.columns = [f"filetype_{f}" for f in top_fileextensions]


# In[96]:


# compute correlation with ttm
corr = filetype_df.corrwith(feature_df["ttm_class"])
corr.sort_values(ascending=False)


# In[97]:


corr[corr.abs() > 0.01]


# In[98]:


# update list of features to keep for training
# take everything that has more than 0.01 correlation magnitude
keep_cols = corr[corr.abs() > 0.01].index.tolist()
selected_features += keep_cols

# only keep the relevant columns now
filetype_df = filetype_df[keep_cols]


# In[99]:


# explode the list column so that there is 1 column representing frequency of each word
title_wordcount_df = pd.DataFrame(feature_df.title_word_counts_vec.to_list())

title_wordcount_df.index = feature_df.index
title_wordcount_df.columns = [f"title_wordcount_{w}" for w in unique_words]


# In[100]:


# compute correlation with ttm
corr = title_wordcount_df.corrwith(feature_df["ttm_class"])
corr = corr.dropna()
corr.sort_values(ascending=False)


# In[101]:


corr[corr.abs() > 0.04]


# In[102]:


# update list of features to keep for training
# take everything that has more than 0.01 correlation magnitude
keep_cols = corr[corr.abs() > 0.04].index.tolist()
selected_features += keep_cols

title_wordcount_df = title_wordcount_df[keep_cols]


# Using various feature importance scores such as chi-squared, mutual information statistic, correlation values, we have selected the most relevant features amongst all of our features. 

# In[103]:


selected_features


# ## Save Feature Engineered Dataset

# In[104]:


# join all types of features
ttm_dataset = pd.concat(
    [feature_df, filetype_df, title_wordcount_df], axis=1, ignore_index=False
)

# keep only the selected features and the prediction label
ttm_dataset = ttm_dataset[selected_features + ["ttm_class", "time_to_merge"]]

ttm_dataset.head()


# In[105]:


ttm_dataset.info()


# ## Save results to Ceph or locally
# 
# Using the helper function to save the data frame in a parquet format on the Ceph bucket if we are running in automation, and locally if not.

# In[106]:


## Sanity check to see if the dataset is the same

if REMOTE:
    cc = CephCommunication(s3_endpoint_url, s3_access_key, s3_secret_key, s3_bucket)
    try:
        cc.upload_to_ceph(
            ttm_dataset,
            s3_path,
            "ttm_dataset.parquet",
        )
    except ValueError as ve:
        print(ve)
        print("Files already uploaded to S3")

else:
    save_to_disk(
        ttm_dataset,
        OUTPUT_DATA_PATH,
        "ttm_dataset.parquet",
    )


# # Conlusion
# 
# This notebook walked through how to access the relevant PR data for the OpenShift Origin project we whish to analyze along with the rest of the related CI data in this project. We also outlined what is contained in each record and provided a quick demonstration of how to convert the mixed data types into numerical values for statistical analysis and eventually machine learning. 
# 
# We have also engineered some possible features which can be used to predict time to merge of a PR. 
# 
# 
# - time_to_merge: time taken to merge a PR in seconds, this is feature we will try to predict
# - changed_files_number: integer indicating the number of files changed, this is an available column
# - body_size: integer indicating the number of words in the body of the PR, derived from body
# - size: integer corresponding to a categorical label generated by GitHub which summerizes the size of the changes made to the code base 
# - title_word_counts_vec: vector or list indicating how many times each word appeared in the title
# - is_reviewer: categorical variable indicating whether the creator of the PR is a reviewer, derived from created_by and additional repo data
# - is_approver: categorical variable indicating whether the creator of the PR is an approver, derived from created_by and additional repo data
# - num_prev_merged_prs: categorical variable indicating whether the creator of the PR has previously created PRs, derived from created_by
# - created_at_day: categorical variable indicating the day of month the PR has been created, derived from created_at
# - created_at_month: categorical variable indicating the month of the year that the PR has been created, derived from created_at
# - created_at_weekday: categorical variable indicating the day of the week the PR has been created, derived from created_at
# - created_at_hour: categorical variable indicating the hour of the day the PR has been created, derived from created_at
# - commits_number: integer indicating the number of commiyts in the PR
# - changed_file_type_vec:  vector or list consisting of number of files changed for the top 100 most frequently occuring datatypes, derived from changed_files
# - change_in_.github: categorical variable indicating whether the PR changes a file in the .github directory, derived from changed_files
# - change_in_cmd: categorical variable indicating whether the PR changes a file in the cmd directory, derived from changed_files 
# - change_in_docs: categorical variable indicating whether the PR changes a file in the docs directory, derived from changed_files 
# - change_in_e2echart: categorical variable indicating whether the PR changes a file in the e2echart directory, derived from changed_files 
# - change_in_examples: categorical variable indicating whether the PR changes a file in the examples directory, derived from changed_files 
# - change_in_hack: categorical variable indicating whether the PR changes a file in the hack directory, derived from changed_files 
# - change_in_images: categorical variable indicating whether the PR changes a file in the images directory, derived from changed_files 
# - change_in_pkg: categorical variable indicating whether the PR changes a file in the pkg directory, derived from changed_files 
# - change_in_test: categorical variable indicating whether the PR changes a file in the test directory, derived from changed_files 
# - change_in_tools: categorical variable indicating whether the PR changes a file in the tools directory, derived from changed_files 
# - change_in_vendor: categorical variable indicating whether the PR changes a file in the vendor directory, derived from changed_files 
# - change_in_root: categorical variable indicating whether the PR changes a file in the root directory of the repo, derived from changed_files 
# 
# As next steps, we will train a classifier to predict time to merge of a PR using the features.
