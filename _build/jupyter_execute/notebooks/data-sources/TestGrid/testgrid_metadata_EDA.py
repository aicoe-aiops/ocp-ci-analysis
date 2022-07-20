#!/usr/bin/env python
# coding: utf-8

# # TestGrid: exploring in-depth metadata
# 
# 
# _Note: This notebook can be run and experimented with directly using the "ocp-ci-analysis" image at our [jupyterhub instance](https://jupyterhub-opf-jupyterhub.apps.smaug.na.operate-first.cloud/hub/login) hosted on the MOC._  
# 
# This notebook explores metadata present at a Test level within the existing TestGrid data at [testgrid.k8s.io](testgrid.k8s.io).
# 
# For initial exploration of the Testgrid data visit [testgrid_EDA.ipynb](testgrid_EDA.ipyb).

# In[1]:


import requests
import datetime as dt

from IPython.display import Image


# **What is Testgrid?** According to the project's [readme](https://github.com/GoogleCloudPlatform/testgrid) it is a, "highly configurable, interactive dashboard for viewing your test results in a grid!" In other words, its an aggregation and visualization platform for CI data. Our hope is that this aggregation encodes some of the subject matter experts' knowledge, and will provide better initial features than going straight to the more complex underlying CI data artifacts [here](https://gcsweb-ci.apps.ci.l2s4.p1.openshiftapps.com/gcs/origin-ci-test/logs/). 

# Lets look into the various metadata fields present on a test level within each job for the testgrid dataset.
# 
# To pull the desired json we can follow this syntax: 
# 
# `https://testgrid.k8s.io/{dashboard_name}/table?&show-stale-tests=&tab={job_name}`
# 
# Here we will get the data for 2 dashboard and jobs for further exploration as examples.
# 
# 1. **dashboard** : "redhat-openshift-ocp-release-4.2-informing" ,
#     **job** :  "release-openshift-origin-installer-e2e-aws-upgrade-rollback-4.1-to-4.2" 
# 
# 2. **dashboard** : "jetstack-cert-manager-master" ,
#     **job** : "ci-cert-manager-bazel" 

# In[2]:


openshift_testgrid = requests.get(
    "https://testgrid.k8s.io/redhat-openshift-ocp-release-4.2-informing/table? \
    &show-stale-tests=&tab=release-openshift-origin-installer-e2e-aws-upgrade-rollback-4.1-to-4.2"
)


# In[3]:


jetstack_testgrid = requests.get(
    "https://testgrid.k8s.io/jetstack-cert-manager-master/table? \
    &show-stale-tests=&tab=ci-cert-manager-bazel"
)


# `openshift_testgrid` has all the fields expected by TestGrid's javascript client. It represents a grid of test results, with additional annotations for metadata.

# The first thing we want to do is see what what keys we have in our detailed job data. Below we can see that there are a number of fields in this json.

# In[4]:


for i in openshift_testgrid.json().keys():
    print(i)


# We will now explore each metadata within the json and try to understand the significance of the values it contains

# The summary field is always empty, hence we will ignore that here.

# ## test-group-name

# In[5]:


print(openshift_testgrid.json()["test-group-name"])


# As we can see above, the`test-group-name` is a string value which contains the particular job name for which all the tests are executed. 

# ## query

# In[6]:


print(openshift_testgrid.json()["query"])


# The above `query` is a string value which contains the `gcs_prefix` pertaining to the job. The `gcs_prefix` is the path to the test data stored in gcs (some-bucket/some/optional/path). The metadata can also be seen [here](https://testgrid.k8s.io/q/config/redhat-openshift-ocp-release-4.2-informing/release-openshift-origin-installer-e2e-aws-upgrade-rollback-4.1-to-4.2) and is documented [here](https://github.com/GoogleCloudPlatform/testgrid/blob/master/pb/config/config.proto).

# ## status

# In[7]:


print(openshift_testgrid.json()["status"])


# The `status` aggregates the tests for which a result - Pass, Fail, Flaky, Running was displayed and the time taken to serve the status. The time taken is calculated by aggregating times for different phases, which are further broken down in `phase-timer`. The status line is displayed on the Status bar for the job.

# ## phase-timer

# In[8]:


print(openshift_testgrid.json()["phase-timer"])


# The `phase-timer` is a dictionary that contains the duration of each update cycle and stores the time taken for different phases. The aggregate time is used to populate the `status` displayed on the status bar of the job.

# ## cached

# In[9]:


print(openshift_testgrid.json()["cached"])


# `cached` holds a True or False value to determine if the response is loaded from a cache.

# ## bugs

# In[10]:


print(openshift_testgrid.json()["bugs"])


# `bugs` is an empty dictionary here, but it should contain any linked bugs created for the job

# ## changelists

# In[11]:


print(openshift_testgrid.json()["changelists"])


# `changelists` list consisting of IDs for each test within a job. The `changelists` can be used in a URL template to visit the Prow Dashboard after clicking on a cell such as `https://prow.k8s.io/view/gcs/<gcs_prefix>/<changelists>` for eg. https://prow.ci.openshift.org/view/gcs/origin-ci-test/logs/release-openshift-origin-installer-e2e-aws-upgrade-rollback-4.1-to-4.2/1365457253460611072

# ## column_ids

# In[12]:


print(openshift_testgrid.json()["column_ids"])


# In[13]:


len(openshift_testgrid.json()["column_ids"])


# In[14]:


print("\ue0001369630643398381568")


# In[15]:


ord("\ue000")


# `column_ids` are timestamps from `changelists` preceded by a unicode character `\ue000`.

# ## custom-columns

# In[16]:


print(openshift_testgrid.json()["custom-columns"])


# In[17]:


print(jetstack_testgrid.json()["custom-columns"])


# For the first dashboard, `custom-columns` did not contain any values, but for the second dashboard, we see that `custom-columns` is nothing but a list of column headers with commit-ids for each test.
# 
# For example, as we see below '9f343ec58' is common header and commit-id for the first tests.

# In[18]:


Image(filename="../../../docs/assets/images/metadata-3.png")


# In[19]:


len(openshift_testgrid.json()["custom-columns"][0])


# ## column-header-names

# In[20]:


print(openshift_testgrid.json()["column-header-names"])


# This is not relevant to our current dashboards which we are looking at, since the values for these column headers are vacant as seen below.

# In[21]:


Image(filename="../../../docs/assets/images/metadata-1.png")


# However, in other dashboards like https://testgrid.k8s.io/jetstack-cert-manager-master#ci-cert-manager-bazel, these consist of 'Date', 'Time', 'Build Number', 'Commit', 'Infra-Commit' for each Test. 
# 

# In[22]:


Image(filename="../../../docs/assets/images/metadata-2.png")


# ## groups

# In[23]:


print(jetstack_testgrid.json()["groups"])


# Not sure what this means

# ## metrics

# In[24]:


print(jetstack_testgrid.json()["metrics"])


# `metrics` is a list consisting of string labels for viewing additional metrics such as `test-duration-minutes`

# In[25]:


Image(filename="../../../docs/assets/images/metadata-4.png")


# ## row_ids

# In[26]:


print(openshift_testgrid.json()["row_ids"])


# `row_ids` is a list consisting of all the various tests executed for the job `release-openshift-origin-installer-e2e-aws-upgrade-rollback-4.1-to-4.2`. Each job also has an `Overall` row which summarises all the test runs for the job.

# In[27]:


len(openshift_testgrid.json()["row_ids"])


# We see that as of right now, `43` tests were run for the job `release-openshift-origin-installer-e2e-aws-upgrade-rollback-4.1-to-4.2` in the dashboard `redhat-openshift-ocp-release-4.2-informing`

# In[28]:


print(jetstack_testgrid.json()["row_ids"])


# In[29]:


len(jetstack_testgrid.json()["row_ids"])


# Similary, we see the various tests executed for the job `ci-cert-manager-bazel` in dashboard `jetstack-cert-manager-master`.

# ## timestamps

# In[30]:


print(openshift_testgrid.json()["timestamps"])


# `timestamps` returns the timestamp of when the test runs were executed.

# In[31]:


len(openshift_testgrid.json()["timestamps"])


# In[32]:


t1 = openshift_testgrid.json()["timestamps"]
dates = [dt.datetime.fromtimestamp(item / 1000).strftime("%d-%m-%y") for item in t1]
dates


# We can convert them to datetime format for further analysis.

# In[33]:


print(jetstack_testgrid.json()["timestamps"])


# In[34]:


len(jetstack_testgrid.json()["timestamps"])


# ## clusters

# In[35]:


print(openshift_testgrid.json()["clusters"])


# In[36]:


print(jetstack_testgrid.json()["clusters"])


# Not very sure what `clusters` should return. According to: https://github.com/GoogleCloudPlatform/testgrid/blob/ce1b1405fc889f371fa047ac03adfab23447c804/README.md#clustered-failures, `clusters` should probably return the clustered failure tests. However, this does not seem to be the case since for `release-openshift-origin-installer-e2e-aws-upgrade-rollback-4.1-to-4.2` in the dashboard `redhat-openshift-ocp-release-4.2-informing`, we see that when we filter the dashboard tab to "Display Clustered Failure Tests" it returns the following:

# In[37]:


Image(filename="../../../docs/assets/images/testgrid_02.png")


# ## test_id_map

# In[38]:


print(openshift_testgrid.json()["test_id_map"])


# In[39]:


print(jetstack_testgrid.json()["test_id_map"])


# This seems to be a lookup map for shortening long test IDs in each row: https://github.com/GoogleCloudPlatform/testgrid/blob/master/pb/response/types.proto#L133

# ## test-metadata

# In[40]:


print(openshift_testgrid.json()["test-metadata"])


# In[41]:


print(jetstack_testgrid.json()["test-metadata"])


# ## stale-test-threshold

# In[42]:


print(openshift_testgrid.json()["stale-test-threshold"])


# In[43]:


print(jetstack_testgrid.json()["stale-test-threshold"])


# `stale-test-threshold` probably means the number of times a test is _not_ run for a particular job, before it gets labeled as "stale" and hidden in the UI. For both the jobs in this example, this threshold is 10.

# ## num-stale-tests

# In[44]:


print(openshift_testgrid.json()["num-stale-tests"])


# We see that for job`release-openshift-origin-installer-e2e-aws-upgrade-rollback-4.1-to-4.2` in the dashboard `redhat-openshift-ocp-release-4.2-informing`, the number of stale tests is `0`.

# In[45]:


print(jetstack_testgrid.json()["num-stale-tests"])


# Similary, for the job `ci-cert-manager-bazel` in dashboard `jetstack-cert-manager-master`, the number of stale tests were `0`.

# ## add-tabular-name-option

# In[46]:


print(openshift_testgrid.json()["add-tabular-names-option"])


# In[47]:


print(jetstack_testgrid.json()["add-tabular-names-option"])


# `add-tabular-names-option` seems to return a True/False value, indicating whether additional tabular names can be added.

# ## show-tabular-names

# In[48]:


print(openshift_testgrid.json()["show-tabular-names"])


# In[49]:


print(jetstack_testgrid.json()["show-tabular-names"])


# `show-tabular-names` also returns a True/False value, but doesn't seem too relevant for our analysis.

# ## description

# In[50]:


print(openshift_testgrid.json()["description"])


# In[51]:


print(jetstack_testgrid.json()["description"])


# The `description` field seems to be a general description of the job. However, most of them seem to return empty values.

# ## bug-component

# In[52]:


print(openshift_testgrid.json()["bug-component"])


# In[53]:


print(jetstack_testgrid.json()["bug-component"])


# Not too sure what this should return.

# ## code-search-path

# In[54]:


print(openshift_testgrid.json()["code-search-path"])


# In[55]:


print(jetstack_testgrid.json()["code-search-path"])


# The `code-search-path` seems to return the relevant GitHub repository path related to the job and the test runs executed.

# ## overall-status

# The `overall-status` field returns a status code describing the entire "job" (the individual tabs within a dashboard in TestGrid are jobs). The mapping and definition of the `overall-status` codes are defined here: https://github.com/GoogleCloudPlatform/testgrid/blob/1dbc0e8b6ef997b9c90b3fb79518c35ce4d27ea2/pb/summary/summary.proto#L163
# 
# * 0 = NOT_SET
# * 1 = UNKNOWN
# * 2 = PASS
# * 3 = FAIL
# * 4 = FLAKY
# * 5 = STALE
# * 6 = BROKEN

# In[56]:


print(openshift_testgrid.json()["overall-status"])


# Hence, for the job `release-openshift-origin-installer-e2e-aws-upgrade-rollback-4.1-to-4.2` in the dashboard `redhat-openshift-ocp-release-4.2-informing`, the overall status is `3` which is `Fail`.

# In[57]:


print(jetstack_testgrid.json()["overall-status"])


#  Similarly, the overall status for the job `ci-cert-manager-bazel` in dashboard `jetstack-cert-manager-master` is `3`, which attributes it as `Fail`.

# What is interesting to note is that the `overall-status` field is different from the `Overall` row we see in the TestGrid dashboard. The `overall-status` defines the status on a "job" level, whereas the `Overall` defines the status on a "build" level. 
# 
# For eg: in the dashboard `redhat-openshift-ocp-release-4.2-informing`, under the tab aka job `release-openshift-origin-installer-e2e-aws-upgrade-rollback-4.1-to-4.2` there is an "Overall" row as highlighted below, which defines the overall status for each "build" executed in the job. Each build comprises of multiple tests and the overall status of the build is an aggregate of all its test statuses. 

# In[58]:


Image(filename="../../../docs/assets/images/testgrid_03.png")


# ## about-dashboard-url

# In[59]:


openshift_testgrid.json()["about-dashboard-url"]


# In[60]:


jetstack_testgrid.json()["about-dashboard-url"]


# Not entirely sure, but maybe this field shows the URL which describes this particular dashboard.

# ## results-text

# In[61]:


print(openshift_testgrid.json()["results-text"])


# This field usually just says "see results on prow". If we manually go the testgrid website : https://testgrid.k8s.io/redhat-openshift-ocp-release-4.2-informing#release-openshift-origin-installer-e2e-aws-upgrade-rollback-4.1-to-4.2&show-stale-tests= and click on any test, we will be re-directed to the prow dashboard (https://prow.ci.openshift.org/view/gcs/origin-ci-test/logs/release-openshift-origin-installer-e2e-aws-upgrade-rollback-4.1-to-4.2/1371445677782994944) where we can find these results.

# ## latest-green

# In[62]:


print(openshift_testgrid.json()["latest-green"])


# In[63]:


print(jetstack_testgrid.json()["latest-green"])


# Not too sure what this should be returning. Manually checked this for a few other dashboards/tabs, but all were empty.

# ## triage-enabled

# In[64]:


print(openshift_testgrid.json()["triage-enabled"])


# In[65]:


print(openshift_testgrid.json()["triage-enabled"])


# In addition to the [testgrid](https://testgrid.k8s.io/) dashboard, Google has another dashboard called [triage](https://go.k8s.io/triage). This dashboard "shows clusters of similar test failures across all jobs", and suggests what general subject these failures related to e.g. storage. So this the value, **True** or **False**, returned in this field probably tells whether triage is enabled or not for this particular dashboard/tab.

# ## notifications

# In[66]:


print(openshift_testgrid.json()["notifications"])


# In[67]:


print(jetstack_testgrid.json()["notifications"])


# This value looks like should be returning the notifications if any. For various dashboards and jobs we looked at it however returned **None**

# ## `xyz-template`-type fields
# 
# 
# These fields are [instances](https://github.com/GoogleCloudPlatform/testgrid/blob/master/pb/response/types.proto#L147) of the [LinkTemplate](https://github.com/GoogleCloudPlatform/testgrid/blob/master/pb/config/config.pb.go#L1775) struct on the testgrid repo. These structs/jsons have three keys:
# 
# 1. url
# 2. name
# 3. options
# 
# Together, these keys describe the format of various URLs where additional information regarding the tests can be found, where bugs can be filed, etc.

# ### open-test-template

# In[68]:


openshift_testgrid.json()["open-test-template"]


# In[69]:


jetstack_testgrid.json()["open-test-template"]


# This field seems to be the template for the URL where more details for a particular test can be found. That is, it's the "general format" for the Prow page for each test.

# ### file-bug-template

# In[70]:


openshift_testgrid.json()["file-bug-template"]


# In[71]:


jetstack_testgrid.json()["file-bug-template"]


# This field seems to be the template for the URL where bugs can be filed for the respective organization/dashboard/job.

# ### attach-bug-template

# In[72]:


openshift_testgrid.json()["attach-bug-template"]


# In[73]:


jetstack_testgrid.json()["attach-bug-template"]


# This is probably the template for URL where bugs can be attached. However, it's empty for all of the instances we looked into.

# ### results-url-template

# In[74]:


openshift_testgrid.json()["results-url-template"]


# In[75]:


jetstack_testgrid.json()["results-url-template"]


# This field seems to be the template for the URL where historical results can be viewed. That is, the "Job History" tab on Prow for each test.

# ### code-search-url-template

# In[76]:


openshift_testgrid.json()["code-search-url-template"]


# In[77]:


jetstack_testgrid.json()["code-search-url-template"]


# This field is the template for the URL where you can view diffs / changes across branches (basically to the git "Compare changes" page).

# ### open-bug-template

# In[78]:


openshift_testgrid.json()["open-bug-template"]


# In[79]:


jetstack_testgrid.json()["open-bug-template"]


# This field seems to be the template for the URL that contains the link to issues page for the respective repository, however the name and options look empty.

# ### context-menu-template

# In[80]:


openshift_testgrid.json()["context-menu-template"]


# In[81]:


jetstack_testgrid.json()["context-menu-template"]


# This field is mostly empty for all intances we looked into. So not sure what this URL is supposed to be.

# ## Metadata within the `statuses` field

# In[82]:


print(openshift_testgrid.json()["tests"][0].keys())


# In[83]:


print((jetstack_testgrid.json()["tests"][0].keys()))


# ### linked_bugs

# In[84]:


print((openshift_testgrid.json()["tests"][0])["linked_bugs"])


# In[85]:


print((jetstack_testgrid.json()["tests"][0])["linked_bugs"])


# In[86]:


# is this non-empty for any other test in any other tabs?
print("Checking OpenShift dashboard")
for t in openshift_testgrid.json()["tests"]:
    if len(t["linked_bugs"]):
        print(t["linked_bugs"])

print("Checking non-OpenShift dashboard")
for t in jetstack_testgrid.json()["tests"]:
    if len(t["linked_bugs"]):
        print(t["linked_bugs"])


# Linked bugs is empty.

# ### messages

# In[87]:


print((openshift_testgrid.json()["tests"][0])["messages"])


# In[88]:


print((jetstack_testgrid.json()["tests"][0])["messages"])


# In[89]:


len((jetstack_testgrid.json()["tests"][0])["messages"])


# Messages contains messages for first few tests of openshift testgrid but good information for jetstack testgrid.

# ### short_texts

# In[90]:


print((openshift_testgrid.json()["tests"][0])["short_texts"])


# In[91]:


print((jetstack_testgrid.json()["tests"][0])["short_texts"])


# In[92]:


print((jetstack_testgrid.json()["tests"][1])["short_texts"])


# Not sure what `F` or `!` signify here. No documentation found about what this could signify.

# ### user_property

# In[93]:


print((openshift_testgrid.json()["tests"][0])["user_property"])


# In[94]:


print((jetstack_testgrid.json()["tests"][0])["user_property"])


# In[95]:


# is this non-None for any other test in any other tabs?
print("Checking OpenShift dashboard")
for t in openshift_testgrid.json()["tests"]:
    if t["user_property"] is not None:
        print(t["user_property"])

print("Checking non-OpenShift dashboard")
for t in jetstack_testgrid.json()["tests"]:
    if t["user_property"] is not None:
        print(t["user_property"])


# This field is empty for all tests in all the dashboards we looked at. Therefore we can't determine exactly what this field means. Nonetheless, based off of the comment [here](https://github.com/GoogleCloudPlatform/testgrid/blob/1dbc0e8b6ef997b9c90b3fb79518c35ce4d27ea2/pb/state/state.pb.go#L537), it seems that `user_property` is any custom, user-defined property that can be associated with a test result.

# ### target

# In[96]:


print((openshift_testgrid.json()["tests"][0])["target"])


# In[97]:


print((jetstack_testgrid.json()["tests"][0])["target"])


# In[98]:


# is "target" same as "name"?
print("Checking OpenShift dashboard")
for t in openshift_testgrid.json()["tests"]:
    if t["target"] != t["name"]:
        print(t["target"])
        print(t["name"])

print("Checking non-OpenShift dashboard")
for t in jetstack_testgrid.json()["tests"]:
    if t["target"] != t["name"]:
        print(t["target"])
        print(t["name"])


# This value in this field is the same as the value in `name` for all tests in all the dashboards we looked at. We couldn't find any strong evidence in the testgrid repo that could explain the difference. Therefore we can't determine exactly how this field is different than test `name`.

# ### original-name

# In[99]:


print((openshift_testgrid.json()["tests"][0])["original-name"])


# In[100]:


print((jetstack_testgrid.json()["tests"][0])["original-name"])


# In[101]:


# is "original-name" same as "name"?
print("Checking OpenShift dashboard")
for t in openshift_testgrid.json()["tests"]:
    if t["original-name"] != t["name"]:
        print(t["original-name"])
        print(t["name"])

print("Checking non-OpenShift dashboard")
for t in jetstack_testgrid.json()["tests"]:
    if t["original-name"] != t["name"]:
        print(t["original-name"])
        print(t["name"])


# This value in this field is the same as the value in `name` for all tests in all the dashboards we looked at. We couldn't find any strong evidence in the testgrid repo that could explain the difference. Therefore we can't determine exactly how this field is different than test `name`

# # Conclusion
# 
# In this notebook, we explored all the metadata fields in the testgrid json data. From the results above, it seems that not all metadata is created equal and that some fields are more useful for data analysis than the others. Specifically, the following fields seem to have more valueable information:
# 
# - `query`: The "gcs_prefix" value here can be used for contstructing prow URL for each test
# - `changelists`: The "ids" value here can be used for contstructing prow URL for each test
# - `custom-column`: Although not properly populated for OpenShift dashboards, in general the values here show the commit ids corresponding to each test run. These can later be used for other types of analysis such as determining what code resulted in passing tests vs failing test, etc.
# - `metrics`: The "test-duration-minutes" value here is useful
# - `row_ids`: This provides the list of test names. Maybe this can be used in the labelwise_encode function (instead of getting test names in a for loop, we can directly set test names to be the value in this field)
# - `timestamps`: Run timestamps
# - `clusters`: Seems useful because it provides similar tests, but it's not heavily populated for OpenShift dashboards
# - `num-stale-tests`: Gives an idea of "health" of test runs (are all tests getting run? how many have not been run in a long time?)
# - `stale-test-threshold`: Might be useful, as it gives context for how severe of a job health indicator the number above is
# - `code-search-path`: Provides the GitHub link for diff'ing across PRs/commits
# - `overall-status`: Gives job summary
# - `open-test-template`: The template here can be used with query and changelists values to get relevant URLs
# - `file-bug-template`: The template here can be used with query and changelists values to get relevant URLs
# - `open-bug-template`: The template here can be used with query and changelists values to get relevant URLs
# - `results-url-template`: The template here can be used with query and changelists values to get relevant URLs
# - `code-search-url-template`: The template here can be used with query and changelists values to get relevant URLs
# - `messages`: Contains some error messages from running the test, but not very densely populated.
# 
# 
# Lastly, the following fields seem to be either noise or empty or repetiions of data already included in other fields.
# 
# - linked-bugs
#     - NOTE: Maybe this could have been a very useful field, but it was found to be always empty. 
# - bugs
#     - NOTE: Maybe this could have been a very useful field, but it was found to be always empty. 
# - cached
# - phase-timer
# - test-group-name
# - status
# - column_ids
# - column-header-names
# - groups
# - test-id-map
# - test-metadata
# - add-tabular-name-option
# - show-tabular-names
# - description
# - bug-component
# - about-dashboard-url
# - results-text
# - latest-green
# - triage-enabled
# - notifications
# - short_texts
# - attach-bug-template
# - context-menu-template
# - target
# - original-name
# - user_property
