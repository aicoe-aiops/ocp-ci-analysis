#!/usr/bin/env python
# coding: utf-8

# # Thoth-station Org Data Extraction

# In this notebook, we will fetch Github PR data for the [thoth-station](https://github.com/thoth-station) reporitories using the [MI tool](https://github.com/thoth-station/mi), pre-process the raw data into suitable dataframes and store them as csv file in s3 bucket.

# In[2]:


import os
import yaml
import requests
from dotenv import find_dotenv, load_dotenv
import warnings
from os.path import join as ospj
from tqdm import tqdm
import boto3
import glob

from github import Github
import pandas as pd
import srcopsmetrics
import pydriller

warnings.filterwarnings("ignore")
load_dotenv(find_dotenv())


# In[4]:


## Create a .env file on your local with the correct configs
GITHUB_ACCESS_TOKEN = os.getenv("GITHUB_ACCESS_TOKEN")

s3_bucket = os.getenv("S3_BUCKET")
s3_endpoint_url = os.getenv("S3_ENDPOINT")
aws_access_key_id = os.getenv("S3_ACCESS_KEY")
aws_secret_access_key = os.getenv("S3_SECRET_KEY")


# In[8]:


# Note: The GitHub access token needs to be exported before importing the srcopmetrics package (current bug)
from srcopsmetrics.entities.issue import Issue  # noqa: E402
from srcopsmetrics.entities.pull_request import PullRequest  # noqa: E402


# We have made the list of thoth-station repos, which would be significant enough for training TTM model. The [notebooks](https://github.com/aicoe-aiops/ocp-ci-analysis/blob/master/notebooks/data-sources/github/thoth_PR_EDA.ipynb) serves the purpose of filtering out the list of significant repos from [thoth-station](https://github.com/thoth-station) organization.

# In[5]:


# List of repos we will be extracting PR data from
repo_list = [
    "thoth-station/.github",
    "thoth-station/adviser",
    "thoth-station/aicoe-ci-pulp-upload-example",
    "thoth-station/amun-api",
    "thoth-station/amun-client",
    "thoth-station/amun-hwinfo",
    "thoth-station/analyzer",
    "thoth-station/ansible-role-argo-workflows",
    "thoth-station/buildlog-parser",
    "thoth-station/cleanup-job",
    "thoth-station/cli-examples",
    "thoth-station/common",
    "thoth-station/core",
    "thoth-station/cve-update-job",
    "thoth-station/datasets",
    "thoth-station/dependency-monkey",
    "thoth-station/dependency-monkey-zoo",
    "thoth-station/document-sync-job",
    "thoth-station/elyra-resnet",
    "thoth-station/fext",
    "thoth-station/glyph",
    "thoth-station/graph-backup-job",
    "thoth-station/graph-metrics-exporter",
    "thoth-station/graph-refresh-job",
    "thoth-station/graph-sync-job",
    "thoth-station/helm-charts",
    "thoth-station/help",
    "thoth-station/image-pusher",
    "thoth-station/init-job",
    "thoth-station/integration-tests",
    "thoth-station/invectio",
    "thoth-station/investigator",
    "thoth-station/isis-api",
    "thoth-station/jupyter-nbrequirements",
    "thoth-station/jupyter-notebook-s2i",
    "thoth-station/jupyterlab-requirements",
    "thoth-station/jupyternb-build-pipeline",
    "thoth-station/kebechet",
    "thoth-station/lab",
    "thoth-station/license-solver",
    "thoth-station/management-api",
    "thoth-station/messaging",
    "thoth-station/metrics-exporter",
    "thoth-station/mi",
    "thoth-station/mi-scheduler",
    "thoth-station/micropipenv",
    "thoth-station/misc",
    "thoth-station/moldavite-api",
    "thoth-station/nepthys",
    "thoth-station/notebooks",
    "thoth-station/package-extract",
    "thoth-station/package-releases-job",
    "thoth-station/package-update-job",
    "thoth-station/performance",
    "thoth-station/pipeline-helpers",
    "thoth-station/pipelines-catalog",
    "thoth-station/pipenv",
    "thoth-station/prescription-sync-job",
    "thoth-station/prescriptions",
    "thoth-station/prescriptions-gh-release-notes-job",
    "thoth-station/prescriptions-refresh-job",
    "thoth-station/ps-cv",
    "thoth-station/ps-ip",
    "thoth-station/ps-nlp",
    "thoth-station/pulp-metrics-exporter",
    "thoth-station/pulp-operate-first-web",
    "thoth-station/pulp-pypi-sync-job",
    "thoth-station/purge-job",
    "thoth-station/python",
    "thoth-station/qeb-hwt",
    "thoth-station/rapidsai-build",
    "thoth-station/ray-ml-notebook",
    "thoth-station/ray-ml-worker",
    "thoth-station/ray-operator",
    "thoth-station/report-processing",
    "thoth-station/reporter",
    "thoth-station/revsolver",
    "thoth-station/s2i",
    "thoth-station/s2i-example",
    "thoth-station/s2i-generic-data-science-notebook",
    "thoth-station/s2i-minimal-notebook",
    "thoth-station/s2i-pytorch-notebook",
    "thoth-station/s2i-scipy-notebook",
    "thoth-station/s2i-tensorflow-gpu-notebook",
    "thoth-station/s2i-tensorflow-notebook",
    "thoth-station/s2i-thoth",
    "thoth-station/search",
    "thoth-station/search-stage",
    "thoth-station/selinon-api",
    "thoth-station/selinon-worker",
    "thoth-station/sentry-openshift",
    "thoth-station/si-aggregator",
    "thoth-station/si-bandit",
    "thoth-station/si-cloc",
    "thoth-station/sigstore-friends",
    "thoth-station/slo-reporter",
    "thoth-station/socrates",
    "thoth-station/solver",
    "thoth-station/solver-error-classfier",
    "thoth-station/solver-errors-reporter",
    "thoth-station/solver-project-url-job",
    "thoth-station/source-management",
    "thoth-station/srcops-notify-bot",
    "thoth-station/srcops-testing",
    "thoth-station/storages",
    "thoth-station/stub-api",
    "thoth-station/support",
    "thoth-station/sync-job",
    "thoth-station/talks",
    "thoth-station/template-project",
    "thoth-station/tensorflow-build-s2i",
    "thoth-station/tensorflow-release-api",
    "thoth-station/tensorflow-release-job",
    "thoth-station/tensorflow-serving-build",
    "thoth-station/tensorflow-symbols",
    "thoth-station/termial-random",
    "thoth-station/thamos",
    "thoth-station/thoth",
    "thoth-station/thoth-application",
    "thoth-station/thoth-github-action",
    "thoth-station/thoth-ops-infra",
    "thoth-station/thoth-pybench",
    "thoth-station/thoth-station.github.io",
    "thoth-station/thoth-toolbox",
    "thoth-station/user-api",
    "thoth-station/website",
    "thoth-station/workflow-helpers",
    "thoth-station/workflows",
]


# In[6]:


# Number of repos
len(repo_list)


# In[ ]:


#using srcopsmetrics to extract PR data from respective repos
for repo in repo_list:
    org = repo.split('/')[0]
    repo = repo.split('/')[1]
    print(f"--->>Extracting data from {org}/{repo}")
    get_ipython().system('python -m srcopsmetrics.cli -clr $org/$repo -e PullRequest # noqa: E999')


# In[13]:


# Fetching the PR for particular org and repo
def get_pr_metrics(org, repo):
    pr = PullRequest(f"{org}/{repo}")
    pr_df = pr.load_previous_knowledge(is_local=True)
    pr_df = pr_df.reset_index()

    pr_df["org"] = org
    pr_df["repo"] = repo

    return pr_df


# In[ ]:


# Saving the PR data as a csv file individually for each repo
for repo in repo_list:
    org = repo.split("/")[0]
    repo = repo.split("/")[1]
    pr_df = get_pr_metrics(org, repo)
    pr_df.to_csv(f"../../../../data/{org}-{repo}.csv")


# In[ ]:


# merge all files together
data_files = [f for f in glob.glob("../../../../data/*.csv")]
df = pd.DataFrame()

for f in tqdm(data_files):
    df = df.append(
        pd.read_csv(f),
    )


# In[8]:


df.head()


# In[14]:


df.to_csv("thoth_PR_data.csv")


# In[16]:


# Uploading the file in the bucket:
# List of file in bucket
s3 = boto3.client(
    "s3",
    endpoint_url=s3_endpoint_url,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
)

s3.upload_file(Filename="thoth_PR_data.csv", Bucket=s3_bucket, Key="thoth_PR_data.csv")


# Hence, we have successfully uploaded the PR data file extracted from the thoth `repo_list` in the s3_bucket.
