#!/usr/bin/env python
# coding: utf-8

# # GitHub Data from `thoth-station` organization

# In this notebook, we will be accessing github data from thoth-station organization. We will be accessing data from all the repos so that we can later filter them and use the data for training TTM ML model. 
# 
# The motivation is to look out for the list of repos containing non-thoth contribution so that we can use the PRs data from those repos for training our model.

# In[80]:


import os
import time
import pandas as pd

from github import Github, RateLimitExceededException
from tqdm import tqdm
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())


# In[81]:


access_token = os.getenv("ACCESS_TOKEN")
g = Github(access_token)


# Checking the extraction rate is important. In this case, we have a minimum rate limit of 5000 request per hour. Once, your rate limit is over. You need to wait for another hour in order to extract the data. 

# In[146]:


g.rate_limiting


# At first, we are extracting the surface level information like number of repos in thoth-station organization. 

# In[ ]:


project_list = g.get_organization("thoth-station").get_repos()

df_github = pd.DataFrame()
for project in tqdm(project_list):

    g = Github(access_token)
    # print(f"Extracting data from {project}")

    repo = project

    PRs = repo.get_pulls(state="all", base="master")

    all_issues = repo.get_issues(state="all")

    df_github = df_github.append(
        {
            "Full_name": repo.full_name,
            "Forks": repo.forks_count,
            "Stars": repo.stargazers_count,
            "last_updated": repo.updated_at,
            "PR_count": PRs.totalCount,
            "Issue_count": all_issues.totalCount,
        },
        ignore_index=True,
    )


# In[83]:


# df_github.to_csv("df_github.csv") #saving it for future use
df = pd.read_csv("df_github.csv", index_col=0)
df = df_github
df.head()


# In[84]:


print(
    f"The number of repos in thoth-station organization is {df['Full_name'].nunique()}"
)


# In the next step, I will be looping over each repo to extract individual issue information.

# In[ ]:


repos = list(df["Full_name"])
# repos = ['thoth-station/package-analyzer']
df_github3 = pd.DataFrame()
for repo in tqdm(repos):
    while True:
        try:
            g = Github(access_token, retry=3, timeout=5)
            # print(f"Extracting data from {project}")
            repo = g.get_repo(repo)

            all_issues = repo.get_issues(state="all")

            for issue in all_issues:
                while True:
                    try:

                        if issue.pull_request is not None:
                            break
                        df_github3 = df_github3.append(
                            {
                                "Project_ID": repo.id,
                                "Name": repo.name,
                                "Full_name": repo.full_name,
                                "issue_number": issue.number,
                                "owner": issue.user.name,
                                "owner_username": issue.user.login,
                            },
                            ignore_index=True,
                        )
                    except RateLimitExceededException as e:
                        print(e.status)
                        print("Rate limit exceeded")
                        print(g.rate_limiting)
                        time.sleep(300)
                        continue
                    break
        except RateLimitExceededException as e:
            print(e.status)
            print("Rate limit exceeded")
            print(g.rate_limiting)
            time.sleep(300)
            continue
        break
# df_github3.to_csv('df_github3.csv')


# In[85]:


df_repo = pd.read_csv("df_github3.csv", index_col=0)


# In[86]:


df_repo.head(20)


# In[87]:


print(f"Number of repos : {df_repo['Full_name'].nunique()}.")


# In[88]:


print(
    f"We see that there is some difference between original repo number({df['Full_name'].nunique()})"
    f" and the number of repo, after extracting the issue information({df_repo['Full_name'].nunique()})."
    f"The difference is because of the fact that the {179-144} repo does not have any issues opened."
    f"Hence, in our next case we will be excluding those repo where there is no issue opened."
)


# **List of repos with no issues**

# In[89]:


df_repo_with_no_issue = df[
    (df["Full_name"].apply(lambda x: x not in list(df_repo["Full_name"])))
]


# In[90]:


len(df_repo_with_no_issue["Full_name"].unique())


# In the next case, we will be filtering based on,
# 
# - Number of stars for each repo.
# - Repo which are active from last year.
# - Contributions from the non-thoth members.

# ### Filtering based on Stars

# In[92]:


df.describe()


# We are filtering those repos having `stars > 2` for all 179 repos from thoth organization.

# In[99]:


df1 = df[(df["Stars"] > 2)]


# In[100]:


print(
    f"After filtering, we get the number of repos having stars greater than 2 is {df1['Full_name'].nunique()}."
)


# ### Filtering based on activity

# We filter and keep only those repo which were active from last year. Here also, we apply the filter on all 179 repos.

# In[101]:


df2 = df[(df["last_updated"] > "2021-5-1")]


# In[103]:


df2.head()


# In[104]:


print(
    f"The number of repo that were active from last year is {df2['Full_name'].nunique()}."
)


# ### Filtering based on contribution from non-thoth account

# We have the list of thoth members, which includes not only present members but also members from the past,

# In[137]:


thoth_members = [
    "codificat",
    "erikerlandson",
    "fridex",
    "Gkrumbach07",
    "goern",
    "Gregory-Pereira",
    "harshad16",
    "HumairAK",
    "KPostOffice",
    "mayaCostantini",
    "meile18",
    "oindrillac",
    "pacospace",
    "schwesig",
    "sesheta",
    "tumido",
    "xtuchyna",
    "bot",
    "sub-mod",
    "GiorgosKarantonis",
    "CermakM",
    "bjoernh2000",
    "srushtikotak",
    "4n4nd",
    "EldritchJS",
    "sesheta-srcops",
    "Shreyanand",
    "bissenbay",
    "saisankargochhayat",
    "pacospace",
]

strings_to_exclude = "|".join(thoth_members)


# In[138]:


df_repo.head(2)


# In[139]:


df3 = df_repo[~df_repo["owner_username"].str.contains(strings_to_exclude)]


# In[140]:


df3.head(20)


# In[141]:


print(f"The names of non-thoth members are : {df3['owner'].unique()}.")


# After excluding those rows which have information about the contribution from thoth members and bots. We are left with the dataset (df3) which list out the contribution from non-thoth account.

# In[142]:


print(
    f"The number of repos which has contribution from non-thoth account is {df3['Full_name'].nunique()}."
)


# ### Union of three filters

# Lastly, once we have the filtered list of all the repo. In order to get a final list of repos which can be significant for further analysis. We will get the union of all three filtered sets.

# In[143]:


df_union = set(df1["Full_name"]).union(
    set(df2["Full_name"]).union(set(df3["Full_name"]))
)


# In[144]:


print(f"The number of repos that we get is {len(df_union)}")


# **The list of repo which we can consider for further analysis**

# In[145]:


df_union


# # Conclusion

# We listed the number of repos that can be significant for training the TTM model. In the next case, we will extract all the issues and PRs data for the above listed repos.
