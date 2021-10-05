#!/usr/bin/env python
# coding: utf-8

# # Bugzilla Data for CI Tests on Testgrid
# 
# Currently, we are analyzing OpenShift CI test runs based on the raw run results data available on testgrid. However, we also want to analyze our CI process in terms of how many bugs we were able to discover, how severely these bugs impacted the product, how accurately did the tests pinpoint the problematic component, and so on. Additionally, having bug related data for the CI tests will also enable us to measure and track several KPIs.
# 
# Therefore, in this notebook we will connect the two data sources: Bugzilla and Testgrid. First, we will identify which bugs are linked with each failing test. Then, we will get detailed information regarding each of these bugs from Red Hat Bugzilla.

# In[1]:


import sys
import requests
import datetime as dt
from io import StringIO
import multiprocessing as mp
from bs4 import BeautifulSoup

from tqdm import tqdm
from wordcloud import WordCloud
from dotenv import load_dotenv, find_dotenv

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import bugzilla

sys.path.insert(1, "../TestGrid/metrics")
from ipynb.fs.defs.metric_template import save_to_disk  # noqa: E402


# In[2]:


# load env vars
load_dotenv(find_dotenv())

# tqdm extensions for pandas functions
tqdm.pandas()

# seaborn plot settings
sns.set(rc={"figure.figsize": (15, 5)})


# In[3]:


# current datetime
current_dt = dt.datetime.now(tz=dt.timezone.utc)


# In[4]:


# get the red hat dashboard names
response = requests.get(
    "https://testgrid.k8s.io/redhat-openshift-informing?id=dashboard-group-bar"
)
html = BeautifulSoup(response.content)
testgrid_script = html.findAll("script")[3]
testgrid_script = testgrid_script.text.split()[5].split(",")
dashboard_names = [x.split(":")[1] for x in testgrid_script if "name" in x]
dashboard_names


# ## Get Linked Bugs

# In this section, we will first identify the linked and associated bugs for all the tests for all jobs under a given dashboard. Then, for the bug ids obtained from this step, we will fetch detailed bug information and better understand the structure and properties of the bugzilla data in the next section. At the end of this section, we'll collect the linked and associated bugs for all tests under each of the jobs displayed on testgrid, and then save this dataset for further analysis in another notebook.
# 
# **NOTE** Running this procedure resulted in really long runtimes: ~30min for one job, >20hrs for all jobs. Therefore we parallelized the code and distributed the workload across multiple processes. This reduced the runtimes to ~1min and ~1hr respectively.

# In[5]:


# manager to share objects across processes
manager = mp.Manager()

# number of max processes
n_max_processes = mp.cpu_count()


# ### Get Jobs under each Dashboard

# In[6]:


# dict where key is dashboard name, value is list of jobs under that dashboard
dashboard_jobs_dict = manager.dict()


def get_jobs_in_dashboard(dj_dict_d_name_tuple):
    """Gets jobs listed under each dashboard.

    :param dj_dict_d_name_tuple: (tuple) Tuple where the first element is the
    shared dict where the result is to be stored and the second element is the
    dashboard name

    NOTE: If we want to have tqdm with a multiprocessing Pool, we need to use
    pool.imap and thus have only one arg passed. Otherwise we can also split
    the args into separate variables
    """
    # unpack args
    dj_dict, d_name = dj_dict_d_name_tuple

    # get list of jobs
    dj_dict[d_name] = tuple(
        requests.get(f"https://testgrid.k8s.io/{d_name}/summary").json().keys()
    )


# list of args to be passed to the function. each process will take one element
# from this list and call the function with it
args = []
for d in dashboard_names:
    args.append(tuple([dashboard_jobs_dict, d]))
args[0]


# In[7]:


# spawn processes and run the function with each arg
with mp.Pool(processes=n_max_processes) as pool:
    _ = list(tqdm(pool.imap(get_jobs_in_dashboard, args), total=len(args)))

# sanity check
dashboard_jobs_dict._getvalue()['"redhat-openshift-ocp-release-4.2-informing"']


# ### Get Tests under each Job

# In[8]:


# dict where key is (dashboard,job), value is list of tests under that job
job_tests_dict = manager.dict()


def get_tests_in_job(jt_dict_dj_pair_tuple):
    """Gets tests run under each job.

    :param jt_dict_dj_pair_tuple: (tuple) Tuple where the first element is the
    shared dict where the result is to be stored and the second element is a
    tuple of (dashboard, job)

    NOTE: If we want to have tqdm with a multiprocessing Pool, we need to use
    pool.imap and thus have only one arg passed. Otherwise we can also split
    the args into separate variables
    """
    # unpack args
    jt_dict, dj_pair = jt_dict_dj_pair_tuple

    # query testgrid for tests in dashboard, job
    ret = requests.get(
        f"https://testgrid.k8s.io/{dj_pair[0]}/table?&show-stale-tests=&tab={dj_pair[1]}"
    )

    # if valid response then add to dict, else print the names to debug
    if ret.status_code == requests.codes.ok:
        jt_dict[dj_pair] = [
            t["name"] for t in ret.json().get("tests")  # , [{'name': None}])
        ]
    else:
        print("non-successful status code for pair", dj_pair)
        jt_dict[dj_pair] = list()


# list of args to be passed to the function. each process will take one element
# from this list and call the function with it
# NOTE: itertools can be used instead of nested for, but this is more readable
args = []
for d, jobs in dashboard_jobs_dict.items():
    for j in jobs:
        args.append(
            tuple(
                [
                    job_tests_dict,  # first arg to function
                    (d, j),  # second arg to function
                ]
            )
        )
args[0]


# In[9]:


# spawn processes and run the function with each arg
with mp.Pool(processes=n_max_processes) as pool:
    _ = list(tqdm(pool.imap(get_tests_in_job, args), total=len(args)))

# sanity check
job_tests_dict._getvalue()[
    (
        '"redhat-openshift-ocp-release-4.2-informing"',
        "periodic-ci-openshift-release-master-ci-4.2-e2e-gcp",
    )
]


# ### Get Linked Bugs under each Test for a Given Dashboard

# In[10]:


# get bugs linked at timestamps up to this amount of time before today
max_age = "336h"

# ci details search url
url = "https://search.ci.openshift.org/"

sample_dashboard = '"redhat-openshift-ocp-release-4.2-informing"'


# In[11]:


# dict where key is (dashboard, job, test), value is list of related bugs
djt_linked_bugs_dict = manager.dict()


def get_bugs_in_test(args_tuple):
    """Gets linked and associated bugs for each test+job.

    Queries the search.ci.openshift website just like the sippy setup does in
    its findBug function defined here:
    https://github.com/openshift/sippy/blob/1a44268082fc600d69771f95a96b4132c9b84285/pkg/buganalysis/cache.go#L230

    :param args_tuple: (tuple) Tuple where the first element is the
    shared dict where the result is to be stored and the second element is a
    tuple of (dashboard, job, test)

    NOTE: If we want to have tqdm with a multiprocessing Pool, we need to use
    pool.imap and thus have only one arg passed. Otherwise we can also split
    the args into separate variables
    """
    # unpack
    djt_linked_bugs, djt_tuple = args_tuple

    # search for linked and associated bugs for this test
    # DO NOT AJAX,MOBILE. THIS HACK PREVENTS REQUEST TIME OUT.
    # read more here - https://stackoverflow.com/a/63377265/9743348
    response = requests.post(
        "https://search.ci.openshift.org/",
        data={
            "type": "bug+junit",
            "context": "-1",
            "name": djt_tuple[1],
            "maxAge": "336h",
            "ajax": "true",
            "mobile": "false",
            "search": djt_tuple[2]
            .split(".", maxsplit=1)[-1]
            .replace("[", r"\[")
            .replace("]", r"\]"),
        },
    )
    soup = BeautifulSoup(response.content)

    # the "em" objects in soup have information that can tell us
    # whether or not this test had a linked bug for the given job name
    em_objects = soup.find_all("em")
    pct_affected = 0
    for em in em_objects:
        if "Found" in em.text:
            pct_affected = float(em.text.split()[2][:-1])
            break

    # init to empty for this test result / reset to empty from previous test result
    test_bugs = []

    # if percent jobs affected is 0 then the linked bugs correspond to another job
    if pct_affected > 0:
        result_rows = soup.find("table").find("tbody").find_all("tr")
        for row in result_rows:
            column_values = row.find_all("td")

            # if there is only 1 column then the result is a junit, not bug
            if len(column_values) > 1:
                # check the second column to make sure its a bug, not junit details
                if column_values[1].text == "bug":
                    test_bugs.append(column_values[0].text[1:])

    djt_linked_bugs[djt_tuple] = test_bugs


# list of args to be passed to the function. each process will take one element
# from this list and call the function with it
# NOTE: this double for loop can be done via itertools too but this is more readable
args = []
for djpair, tests in job_tests_dict.items():
    if djpair[0] == sample_dashboard:
        for t in tests:
            args.append(
                tuple(
                    [
                        djt_linked_bugs_dict,  # first arg to function
                        (*djpair, t),  # second arg to function
                    ]
                )
            )
args[0]


# In[12]:


# spawn processes and run the function with each arg
with mp.Pool(processes=n_max_processes) as pool:
    _ = list(tqdm(pool.imap(get_bugs_in_test, args), total=len(args)))

# sanity check
djt_linked_bugs_dict._getvalue()[
    (
        '"redhat-openshift-ocp-release-4.2-informing"',
        "periodic-ci-openshift-release-master-ci-4.2-e2e-aws-sdn-multitenant",
        "Operator results.operator conditions monitoring",
    )
]


# In[13]:


# set of ALL bugs observed for this dashboard
all_bugs = set()

# flattened list. each element is (dashboard, job, test, list-of-bugs)
djt_linked_bugs_list = []
for k, v in djt_linked_bugs_dict.items():

    djt_linked_bugs_list.append(tuple([*k, v]))
    all_bugs.update(v)

# convert results to df
linked_bugs_df = pd.DataFrame(
    djt_linked_bugs_list, columns=["dashboard", "job", "test_name", "bug_ids"]
)

# drop rows where there are no linked bugs
has_linked_bugs = linked_bugs_df["bug_ids"].apply(len) > 0
print(
    f"Out of {len(has_linked_bugs)} rows, {has_linked_bugs.sum()} had non-empty linked bugs"
)
linked_bugs_df = linked_bugs_df[has_linked_bugs]

linked_bugs_df.head()


# ### Get Linked Bugs under each Test for All Dashboards

# In[14]:


# init as empty dict. key is (dashboard, job, test) and value is the list of related bugs
djt_linked_bugs_dict = manager.dict()

# list of args to be passed to the function
# this time, we will get linked bugs for all tests in ALL dashboards, not just one
args = []
for djpair, tests in job_tests_dict.items():
    for t in tests:
        args.append(
            tuple(
                [
                    djt_linked_bugs_dict,  # first arg to function
                    (*djpair, t),  # second arg to function
                ]
            )
        )

# spawn processes and run the function with each arg
with mp.Pool(processes=n_max_processes) as pool:
    _ = list(tqdm(pool.imap(get_bugs_in_test, args), total=len(args)))

# flattened list. each element is (dashboard, job, test, list-of-bugs)
djt_linked_bugs_list = []
for k, v in djt_linked_bugs_dict.items():
    djt_linked_bugs_list.append(tuple([*k, v]))

# convert results to df
linked_bugs_df = pd.DataFrame(
    djt_linked_bugs_list, columns=["dashboard", "job", "test_name", "bug_ids"]
)


# In[20]:


# drop rows where there are no linked bugs
has_linked_bugs = linked_bugs_df["bug_ids"].apply(len) > 0
print(
    f"Out of {len(has_linked_bugs)} rows, {has_linked_bugs.sum()} had non-empty linked bugs"
)
linked_bugs_df = linked_bugs_df[has_linked_bugs]

# save df
save_to_disk(
    linked_bugs_df,
    "../../../data/raw/",
    f"linked-bugs-{current_dt.year}-{current_dt.month}-{current_dt.day}.parquet",
)


# ## Get Bugzilla Details
# 
# In this section, we will get details for the bug ids collected for the sample dashboard in the above section. We will fetch all the available metadata fields for each bug, and but only explore the values in some of these fields. We will perform the meticulous exploratory analysis for all of the available Bugzilla fields in a future notebook. 

# In[15]:


# connector object to talk to bugzilla
bzapi = bugzilla.Bugzilla("bugzilla.redhat.com")

# look at a sample bug - what properties does this object have?
samplebug = bzapi.getbug(1883345)
vars(samplebug).keys()


# **NOTE** The above shows what fields/properties related to each bugzilla we have available.
# Upon a bit of investigating we found that
# - `_rawdata` just contains the information already captured in other fields in a json format, and thus is redundant
# - `bugzilla` attribute is depracated / old representation used in the python-bugzilla library, and thus is not useful for analysis
# - `_aliases` is a mapping of synonyms for some of the fields, and thus is not useful for analysis
# - The following properties didn't exist for most bugs (it's not that these properties has empty values, it's that the properties themselves didn't exist as a field for most objects of the Bugzilla class):
#     - `qa_contact_detail`
#     - `cf_last_closed`
#     - `cf_clone_of`

# In[16]:


# get all the available fields, except the depracated and duplicate ones
bug_details_to_get = list(vars(samplebug).keys())
bug_details_to_get.remove("_rawdata")
bug_details_to_get.remove("bugzilla")
bug_details_to_get.remove("_aliases")

# these two keys are msissing for a lot of bugs
bug_details_to_get.remove("qa_contact_detail")
bug_details_to_get.remove("cf_last_closed")
bug_details_to_get.remove("cf_clone_of")

bug_details_to_get


# In[17]:


# create a df containing details of all linked and associated bugs
bugs_df = pd.DataFrame(
    columns=["bug_id"] + bug_details_to_get, index=range(len(all_bugs))
)
bugs_df = bugs_df.assign(bug_id=all_bugs)
bugs_df.head()


# In[18]:


def fill_bug_details(bug_row):
    """
    Populate details for each bug
    """
    global bzapi

    try:
        bug = bzapi.getbug(bug_row.bug_id)
    except Exception:
        return bug_row

    for detail in bug_row.index:
        try:
            bug_row[detail] = getattr(bug, detail)
        except AttributeError:
            print(detail)

    return bug_row


bugs_df.progress_apply(fill_bug_details, axis=1)
bugs_df.head()


# In[115]:


# custom converting each column into a dtype that pyarrow can work with is tricky
# as a hack, we'll convert the df to a csv (in a buffer) and then read that csv
# so that pandas does the type comprehension by itself
buffer = StringIO()
bugs_df.to_csv(buffer, index=False)

buffer.seek(0)
bugs_df = pd.read_csv(buffer)

# save raw data
save_to_disk(
    bugs_df,
    "../../../data/raw/",
    f"bug-details-{current_dt.year}-{current_dt.month}-{current_dt.day}.parquet",
)


# ## Inspect Bug Metadata
# 
# In this section, we will look into some of the metadata fields available in bugzilla. We will not go through every field, but rather the ones that seem more important features of a bug.
# 
# To learn more about what each of these fields represents, please check out the official docs at [Bugzilla](https://bugzilla.readthedocs.io/en/latest/using/understanding.html), [Red Hat Bugzilla](https://bugzilla.redhat.com/docs/en/html/using/understanding.html), or [python-bugzilla](https://github.com/python-bugzilla/python-bugzilla/blob/82972796cf04f1ac06525670c272465a66d77da1/man/bugzilla.rst#global-options).

# ### priority
# 
# The priority field is used to prioritize bugs, either by the assignee, or someone else with authority to direct their time such as a project manager.

# In[116]:


vc = bugs_df["priority"].value_counts()
vc


# In[117]:


vc.plot(kind="bar")
plt.xlabel("Priority Label")
plt.ylabel("Number of Bugs")
plt.title("Bug distribution across different Priority Labels")
plt.show()


# ### blocks
# The blocks field lists the bugs that are blocked by this particular bug.

# In[118]:


def get_n_blocked(blockedlist):
    try:
        return len(blockedlist)
    except TypeError:
        return 0


nblocked = bugs_df["blocks"].apply(get_n_blocked)
nblocked.value_counts()


# In[119]:


nblocked.plot(kind="hist", bins=50)
plt.xlabel("Number of Bugs Blocked")
plt.ylabel("Number of Bugs")
plt.title("Bug distribution across different Number of Bugs Blocked")
plt.show()


# ### last_change_time

# In[120]:


last_change_time = pd.to_datetime(bugs_df["last_change_time"])
last_change_time


# In[121]:


last_change_time.hist()
plt.xlabel("Last Change Date")
plt.ylabel("Number of Bugs")
plt.title("Bug distribution across different Last Change Dates")
plt.xticks(rotation=45)
plt.show()


# ### keywords

# In[122]:


bugs_df["keywords"].value_counts()


# In[123]:


# wordcloud to get rough aggregated idea of which keywords occur the most
wordcloud = WordCloud(max_font_size=75, max_words=500).generate(
    bugs_df.keywords.str.cat()
)

# Display the generated image:
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# ### Whiteboard
# A free-form text area for adding short notes and tags to a bug.

# In[124]:


vc = bugs_df["whiteboard"].value_counts()
vc


# In[125]:


vc.plot.bar()
plt.xlabel("Whiteboard text")
plt.ylabel("Number of Bugs")
plt.title("Bug distribution across different Whiteboard texts")
plt.show()


# ### description 
# This conatins descriptions for each bugzilla ticket.

# In[126]:


bugs_df["description"]


# In[127]:


print(bugs_df["description"].iloc[0])


# ### resolution

# In[128]:


vc = bugs_df["resolution"].value_counts()
vc


# In[129]:


vc.plot.bar()
plt.xlabel("Resolution")
plt.ylabel("Number of Bugs")
plt.title("Bug distribution across different Resolutions")
plt.show()


# From the above graph, we can infer that we have most values available for resolution, even though we have many values as empty, this looks like a promising parameter.

# ### cf_doc_type

# In[130]:


vc = bugs_df["cf_doc_type"].value_counts()
vc


# In[131]:


vc.plot.bar()
plt.xlabel("Doc Type")
plt.ylabel("Number of Bugs")
plt.title("Bug distribution across different Doc Types")
plt.show()


# From the above graph, we see that most of the tickets have the value for `doc_type`. This could be used to classify the tickets according to the doc type.

# ### op_sys : Operating Systems

# In[132]:


vc = bugs_df["op_sys"].value_counts()
vc


# In[133]:


vc.plot.bar()
plt.xlabel("Operating System")
plt.ylabel("Number of Bugs")
plt.title("Bug distribution across different Operating Systems")
plt.show()


# From the above graph, we can see that we have four OS(s) across the bugs.

# ### target_release

# In[134]:


bugs_df["target_release"].value_counts().plot.bar()
plt.xlabel("Target Release")
plt.ylabel("Number of Bugs")
plt.title("Bug distribution across different Target Releases")
plt.show()


# From the above graph, we see the various target releases frequency. This value also is mostly not assigned but we still have many observations.

# ### status

# In[135]:


vc = bugs_df["status"].value_counts()
vc


# In[136]:


vc.plot.bar()
plt.xlabel("Status")
plt.ylabel("Number of Bugs")
plt.title("Bug distribution across different Statuses")
plt.show()


# The above graph, shows various status across tickets.

# ### External Bugs

# In[137]:


bugs_df["external_bugs"].value_counts().to_frame().head()


# ### platform
# 
# The `platform` field indicates the hardware platform the bug was observed on.

# In[138]:


vc = bugs_df["platform"].value_counts()
vc


# In[139]:


vc.plot(kind="bar")
plt.xlabel("Platform")
plt.ylabel("Number of Bugs")
plt.title("Bug distribution across different Bug Platforms")
plt.show()


# ### severity
# 
# The `severity` field categorzies the severity level of each bug. Let's see the different severity levels defined. Let's plot a simple graph to visualize the distribution of bug severities

# In[140]:


vc = bugs_df["severity"].value_counts()
vc


# 

# In[141]:


vc.plot(kind="bar")
plt.xlabel("Severity Level")
plt.ylabel("Number of Bugs")
plt.title("Bug distribution across different Bug Severities")
plt.show()


# ### cf_environment
# 
# Not too sure what `cf_environment` is supposed to return

# In[142]:


bugs_df["cf_environment"].value_counts().to_frame()


# ### version
# 
# The `version` field indicates the version of the software the bug was found in. Let's plot a simple graph to visualize the distribution of bugs across different software versions.

# In[143]:


vc = bugs_df["version"].value_counts()
vc


# In[144]:


vc.plot(kind="bar")
plt.ylabel("Number of Bugs")
plt.xlabel("Software Versions")
plt.title("Bug distribution across different Software Versions")
plt.show()


# ### component
# 
# Bugs are categorised into Product and Component. Components are second-level categories and the `component` field indicates which component is affected by the bug.

# In[145]:


vc = bugs_df["component"].value_counts()
vc


# In[146]:


vc.plot(kind="bar")
plt.xlabel("Component")
plt.ylabel("Number of Bugs")
plt.title("Bug distribution across different components")
plt.show()


# ### sub_component
# 
# The `sub_component` field indicates the sub-component of a specifc component the bug affects.

# In[147]:


vc = bugs_df["sub_component"].value_counts()
vc


# In[148]:


vc.plot(kind="bar")
plt.xlabel("Subcomponent")
plt.ylabel("Number of Bugs")
plt.title("Bug distrbution across different subcomponents")
plt.show()


# ### product
# 
# The `product` field indicates the software product affected by the bug.

# In[149]:


vc = bugs_df["product"].value_counts()
vc


# Let's plot a simple graph to visualize the distribution of bugs across different products

# In[150]:


vc.plot(kind="bar")
plt.xlabel("Software Products")
plt.ylabel("Number of Bugs")
plt.title("Bug distrbution across different software products")
plt.show()


# ### fixed_in

# In[151]:


bugs_df["fixed_in"][:15]


# In[152]:


bugs_df["fixed_in"].unique()


# The `fixed_in` field seems to indicate the software version the bug was fixed in. However, it doesn't seem to be applicable to all bugs as some bugs may still be open and not yet resolved.

# ### summary
# 
# The bug summary is a short sentence which succinctly describes what the bug is about.

# In[153]:


bugs_df["summary"]


# In[154]:


print(bugs_df["summary"].iloc[0])


# ### is_open

# In[155]:


vc = bugs_df["is_open"].value_counts()
vc


# In[156]:


vc.plot.bar()
plt.xlabel("is_open")
plt.ylabel("Number of Bugs")
plt.title("Bug distribution across different is_open values")
plt.show()


# ## Contact Metadata
# 
# These fields contain information for people responsible for QA, creation of bug, etc. These are not useful for the initial EDA.
# 
# - `docs_contact_value` and `qa_contact`: The people responsible for contacting and fixing the bug.
# - `creator`: The person who created the bug.
# - `assigned_to`: The person responsible for fixing the bug.
# - `cc`: The mailing list subscribed to get updates for a bug.

# ## Non Useful Metadata
# 
# These fields mostly had either the same value or empty. Therefore, these are not useful for our analysis.
# 
# - `tags`: The tags field seems to be empty for most bugs so we can probably ignore this field.
# - `flags`: The flags field seems to return empty for most bugs. For thos bugs which have this field set, it seems to have redundant information which are already available in other bug fields so we can probably ignore this field.
# - `is_creator_accessible`: The is_creator_accessible field returns a boolean value, but doesn't seem to be useful for our analysis.
# - `cf_release_notes`: The cf_release_notes is the basis of the errata or release note for the bug. It can also be used for change logs. However, it seems to be empty for most bugs and can be excluded from our analysis.
# - `target_milestone`: The target_milestone is used to define when the engineer the bug is assigned to expects to fix it. However, it doesn't seem to be applicable for most bugs.
# - `is_confirmed`: The is_confirmed field seems to return a boolean value (not sure what it indicates) and doesn't seem to be useful for our analysis.
# - `components`: The components field returns the same values as the component field, but in a list format.
# - `sub_components` - The sub_components field is similar to the sub_component field, but returns both the component and sub-component affected by the bug in a dictionary format.
# - `versions`: The versions field returns the same values as the version field, but in a list format.

# # Conclusion
# 
# In this notebook, we show how the bug ids related to each test under each job for all dashboards can be determined, and saved a sample of this mapping as the linked-bugs dataset. We also showed how detailed information for a set of bug ids can be collected, and saved a sample of this dataset as the bug-details dataset. These datasets open up several avenues for exploration, such as in-depth bugs data EDA, and EDA for testgrid + bugs datasets put together, which we will explore in future notebooks.
