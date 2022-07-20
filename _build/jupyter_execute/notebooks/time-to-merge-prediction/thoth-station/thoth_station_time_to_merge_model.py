#!/usr/bin/env python
# coding: utf-8

# # Predicting Time to Merge of a Pull Request
# 
# 
# In this notebook, we train a time to merge model for the `thoth-station` organization pull request data. For that, we will first create transformer objects (based on the explorations in the previous notebook) to extract features from raw PR data. Then, we will train machine learning models to classify a PR's `time_to_merge` into one of the above 10 bins (or "classes").
# 
# Class 0 : < 1 min  
# Class 1 : < 2 mins   
# Class 2 : < 8 mins    
# Class 3 : < 20 mins  
# Class 4 : < 1 hr  
# Class 5 : < 4 hrs  
# Class 6 : < 18 hrs  
# Class 7 : < 3 days  
# Class 8 : < 21 days  
# Class 9: > 3 weeks  
# 

# In[1]:


import os
import sys
import boto3
import tempfile
from io import StringIO
import datetime

import joblib
import warnings
from io import BytesIO
from copy import deepcopy

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer, OrdinalEncoder

from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from dotenv import load_dotenv, find_dotenv

from src.features.build_features import (
    DateTimeDetailsTransformer,
    NumChangedFilesTransformer,
    StringLenTransformer,
    FileTypeCountTransformer,
    TitleWordCountTransformer,
)

metric_template_path = "../../../notebooks/data-sources/TestGrid/metrics"
if metric_template_path not in sys.path:
    sys.path.insert(1, metric_template_path)

from ipynb.fs.defs.metric_template import (  # noqa: E402
    CephCommunication,  # noqa: E402
)  # noqa: E402


# In[2]:


warnings.filterwarnings("ignore")
load_dotenv(find_dotenv())


# ## Get Raw Data

# In[3]:


## CEPH Bucket variables
## Create a .env file on your local with the correct configs,
s3_endpoint_url = os.getenv("S3_ENDPOINT")
s3_access_key = os.getenv("S3_ACCESS_KEY")
s3_secret_key = os.getenv("S3_SECRET_KEY")
s3_bucket = os.getenv("S3_BUCKET")
s3_path = "github/thoth"
REMOTE = os.getenv("REMOTE")
data_path = "../../../data/raw/GitHub/thoth_PR_data.csv"


# In[4]:


if REMOTE:
    print("getting dataset from ceph")
    cc = CephCommunication(s3_endpoint_url, s3_access_key, s3_secret_key, s3_bucket)
    s3_object = cc.s3_resource.Object(s3_bucket, "thoth_PR_data.csv")
    file = s3_object.get()["Body"].read().decode("utf-8")

pr_df = pd.read_csv(StringIO(file))


# In[5]:


pr_df.head()


# In[6]:


# remove PRs from train/test which are still open
pr_df = pr_df[pr_df["closed_at"].notna()]
pr_df = pr_df[pr_df["merged_at"].notna()]


# In[7]:


pr_df["created_at"] = pr_df["created_at"].apply(
    lambda x: int(datetime.datetime.timestamp(pd.to_datetime(x)))
)
pr_df["closed_at"] = pr_df["closed_at"].apply(
    lambda x: int(datetime.datetime.timestamp(pd.to_datetime(x)))
)
pr_df["merged_at"] = pr_df["merged_at"].apply(
    lambda x: int(datetime.datetime.timestamp(pd.to_datetime(x)))
)


# In[8]:


pr_df.head()


# ## Extract Labels from Raw Data

# In[9]:


def get_ttm_class(ttm):
    """
    Assign a ttm "class" / "category" / "bin" to the input numerical ttm value
    E.g. if the time to merge was 1 hours, this function will return
    class "5" which represents "merged in 0.5-2 hours"
    """
    if ttm < 0.00166:
        return 0
    elif ttm < 0.03333:
        return 1
    elif ttm < 0.13333:
        return 2
    elif ttm < 0.33333:
        return 3
    elif ttm < 1:
        return 4
    elif ttm < 4:
        return 5
    elif ttm < 18:
        return 6
    elif ttm < 72:
        return 7
    elif ttm < 504:
        return 8
    else:
        return 9


y = (pr_df["merged_at"] - pr_df["created_at"]).astype("float")
y = y.dropna()
y = (y / 3600).apply(get_ttm_class)
y.head()


# ## Extract Features from Raw Data
# 
# In this section, we will create transformer objects to process raw data as per the feature extraction methods that were found to be the most effective in the previous EDA notebook.
# 
# To ensure that joblib serializes the custom transformer objects correctly, we will write their definitions in [`src/features/build_features.py`](../../src/features/build_features.py) instead of this notebook, and import them from the `src` package. This way, the `src` package can be listed as a dependency wherever these objects need to be deserialized (for example, Seldon server).

# #### size

# In[10]:


# map values to 0,1,2,3,4,5
pr_size_encoder = OrdinalEncoder(categories=[["XS", "S", "M", "L", "XL", "XXL"]])


# #### created_at

# In[11]:


# get day, month, date, etc. from a unix timestamp
dt_details_transf = DateTimeDetailsTransformer()


# #### changed_files_number

# In[12]:


# number of files changed in PR
n_changed_files_transf = NumChangedFilesTransformer()


# #### body_size

# In[13]:


# number of characters in PR description
body_size_transf = StringLenTransformer("body")


# #### filetype_\<type>

# In[14]:


# how many files of the given extension were changed in PR
FILE_EXTENSIONS_TO_COUNT = ["None"]

ftype_count_transf = FileTypeCountTransformer(file_extensions=FILE_EXTENSIONS_TO_COUNT)


# #### title_wordcount_\<word>

# In[15]:


# how many times these words appeared in PR title
WORDS_TO_COUNT = [
    "add",
    "adviser",
    "amun",
    "api",
    "bump",
    "certs",
    "cluster",
    "deployment",
    "environment",
    "exporter",
    "feature",
    "fix",
    "functions",
    "graph",
    "implement",
    "increase",
    "ingestion",
    "introduce",
    "investigator",
    "kustomize",
    "management",
    "middletier",
    "notebook",
    "parallelism",
    "patch",
    "prod",
    "pyproject",
    "python",
    "reporter",
    "required",
    "revert",
    "revsolver",
    "role",
    "s",
    "scorecards",
    "secrets",
    "services",
    "slo",
    "stage",
    "toml",
    "upgrade",
    "v0",
    "v1",
    "v2021",
    "version",
    "wip",
    "💊",
]

title_wc_transf = TitleWordCountTransformer(words=WORDS_TO_COUNT)


# ## Apply Transforms

# In[16]:


# FIXME: breaks if not sorted. can we shave off some runtime by not requiring to sort?
pr_df = pr_df.sort_values(by="created_at")


# In[17]:


pr_df


# In[18]:


# transformer objects compiled into one columntransformer
raw_data_processor = ColumnTransformer(
    [
        ("pr_size", pr_size_encoder, ["size"]),  # 1 cols generated
        ("created_at_details", dt_details_transf, ["created_at"]),  # 4 cols generated
        (
            "n_changed_files",
            n_changed_files_transf,
            ["changed_files_number"],
        ),  # 1 cols generated
        ("body_size", body_size_transf, ["body"]),  # 1 cols generated
        ("n_commits", "passthrough", ["commits_number"]),  # 1 cols generated
        (
            "filetype_counter",
            ftype_count_transf,
            ["changed_files"],
        ),  # 2 cols generated
        ("title_word_counter", title_wc_transf, ["title"]),  # 15 cols generated
    ],
    remainder="drop",
)


# In[19]:


# column names. this is needed because sklearn forcefully converts df to ndarray,
# thus losing column information. this is a hack to retain that info. look for alternatives
cols = (
    [
        "size",
        "created_at_day",
        "created_at_month",
        "created_at_weekday",
        "created_at_hour",
    ]
    + [
        "changed_files_number",
        "body_size",
        "commits_number",
    ]
    + [f"filetype_{f}" for f in FILE_EXTENSIONS_TO_COUNT]
    + [f"title_wordcount_{w}" for w in WORDS_TO_COUNT]
)


# In[20]:


X = raw_data_processor.fit_transform(pr_df)
X = pd.DataFrame(X, index=pr_df.index, columns=cols)
X.head()


# ## Drop NA + Train/Test Split

# In[21]:


# drop entries for which labels are unknown
# also make sure labels and features are consistent
X = X.reindex(y.index)


# In[22]:


# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# In[23]:


# upload X_test and y_test to S3 bucket for testing / running sanity check on the model inference service
cc = CephCommunication(s3_endpoint_url, s3_access_key, s3_secret_key, s3_bucket)

ret = cc.upload_to_ceph(X_test, s3_path, "X_test.parquet")
print(ret["ResponseMetadata"]["HTTPStatusCode"])

ret = cc.upload_to_ceph(y_test.to_frame("ttm_class"), s3_path, "y_test.parquet")
print(ret["ResponseMetadata"]["HTTPStatusCode"])


# In[24]:


# convert from pandas series to lists to avoid warnings during training
y_train = y_train.to_list()
y_test = y_test.to_list()


# ## Scale data

# In[25]:


# lets apply a yeo johnson transform to try to make the data more gaussian
scaler = PowerTransformer()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ## Define Training and Evaluation Pipeline

# Here, we will define a function to train a given classifier on the training set and then evaluate it on the test set. 

# In[26]:


def train_evaluate_classifier(clf, xtrain, ytrain, xtest, ytest):
    # Train our classifier
    clf.fit(xtrain, ytrain)

    # Make predictions
    preds = clf.predict(xtest)

    # View classification report
    print(classification_report(ytest, preds))

    # Plot confusion matrix heatmap
    plt.figure(figsize=(16, 12))
    cf_matrix = confusion_matrix(ytest, preds)
    group_counts = ["{0:0.0f}\n".format(value) for value in cf_matrix.flatten()]
    group_percentages = [
        "{0:.2%}".format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)
    ]
    box_labels = [
        f"{v1}{v2}".strip() for v1, v2 in zip(group_counts, group_percentages)
    ]
    box_labels = np.asarray(box_labels).reshape(cf_matrix.shape[0], cf_matrix.shape[1])

    sns.heatmap(cf_matrix, cmap="OrRd", annot=box_labels, fmt="")
    plt.xlabel("Predicted TTM Label")
    plt.ylabel("True TTM Label")
    plt.title("Confusion Matrix Heatmap")


# ## Define Models and Parameters
# 
# Next, we will define and initialize the classifiers that we will be exploring for the time-to-merge prediction task.

# ### Gaussian Naive Bayes

# In[27]:


# Initialize classifier
gnb = GaussianNB()


# ### SVM

# In[28]:


# Initialize classifier
svc = SVC(random_state=42)


# ### Random Forest

# In[29]:


# Initialize classifier
rf = RandomForestClassifier(
    n_estimators=200,
    max_features=0.75,
    random_state=42,
    n_jobs=-1,
)


# ### XGBoost

# In[30]:


# Initialize classifier
xgbc = XGBClassifier(
    n_estimators=125,
    learning_rate=0.1,
    random_state=42,
    verbosity=1,
    n_jobs=-1,
)


# ## Compare Model Results
# 
# Finally, we will run the train all of the classifiers defined above and evaluate their performance. 

# ### Experiment 1: Train using all features
# 
# First, lets train the classifiers using all the engineered features as input.

# #### Naive Bayes

# In[31]:


train_evaluate_classifier(gnb, X_train_scaled, y_train, X_test_scaled, y_test)


# #### SVM

# In[32]:


train_evaluate_classifier(svc, X_train_scaled, y_train, X_test_scaled, y_test)


# #### Random Forest

# In[33]:


train_evaluate_classifier(rf, X_train_scaled, y_train, X_test_scaled, y_test)


# #### XG Boost

# In[34]:


train_evaluate_classifier(xgbc, X_train_scaled, y_train, X_test_scaled, y_test)


# Based on the results above, it seems the Naive Bayes performs poorly compared to others, amongst SVM, Random Forest and XGBoost, seems like the Random Forest classifier outperforms all others, followed closely by the XGBoost. The Naive bayes seems to be heavily biased towards a few classes. On the contrary, the random forest SVM and the  XGBoost models seem to be less biased and have mis-classifications within the bordering classes amongst the ordinal classes.
# 
# Note that for model deployment (which is the eventual goal), we will also need to include any scaler or preprocessor objects. This is because the input to the inference service will be raw unscaled data. We plan to address this issue by using `sklearn.Pipeline` object to package the preprocessor(s) and model as one "combined" model. Since the random forest performs best here, we will save the random forest as the "best" model here. In the step below, we create a copy of the model so that we can save it to S3 later on and use it for model deployment.

# In[35]:


# create a clone (create a copy of the object with the learned weights)
selected_model = deepcopy(rf)


# In[36]:


# sanity check
print(classification_report(y_test, selected_model.predict(X_test_scaled)))


# ### Experiment 2: Let's train using pruned features
# 
# In the previous notebook we performed some feature engineering and pruned the number of features down. However, it might be possible that further pruning the features based on the importances given to them by the models yields more generalizable and accurate models. So in this section, we will explore using Recursive Feature Elimination (RFE) to rank the features in terms of their importance, and recursively select the best subsets to train our models with.

# In[37]:


# use the random forest classifier as the base estimator since it had the highest f1
selector = RFE(rf, n_features_to_select=20, step=5)
selector = selector.fit(X_train_scaled, y_train)


# In[38]:


# No of top features to select
top = 20


# In[39]:


ranks = selector.ranking_
ranks


# In[40]:


cols = X_train.columns.to_numpy()
cols


# In[41]:


indices_by_ranks = ranks.argsort()
indices_by_ranks


# In[42]:


sorted_ranks = ranks[indices_by_ranks]
sorted_ranks


# In[43]:


cols_by_rank = cols[indices_by_ranks]
cols_by_rank[:top]


# In[44]:


# prune the training set
X_train_scaled_pruned = X_train_scaled[:, selector.support_]
X_test_scaled_pruned = X_test_scaled[:, selector.support_]


# #### Naive Bayes

# In[45]:


train_evaluate_classifier(
    gnb, X_train_scaled_pruned, y_train, X_test_scaled_pruned, y_test
)


# #### SVM

# In[46]:


train_evaluate_classifier(
    svc, X_train_scaled_pruned, y_train, X_test_scaled_pruned, y_test
)


# #### Random Forest

# In[47]:


train_evaluate_classifier(
    rf, X_train_scaled_pruned, y_train, X_test_scaled_pruned, y_test
)


# #### XGBoost

# In[48]:


train_evaluate_classifier(
    xgbc, X_train_scaled_pruned, y_train, X_test_scaled_pruned, y_test
)


# From the confusion matrices above, we can conclude that the models perform slightly better when trained using only the RFE-pruned subset as compared to all the features. The best model(random forest) performs almost the same, so we will keep all the features for now.

# ## Create sklearn Pipeline

# Here, we will create an sklearn pipeline consisting of 2 steps, scaling of the input features and the classifier itself. We will then save this Pipeline as a `model.joblib` file on S3 for serving the model pipeline using the Seldon Sklearn Server.

# In[49]:


pipe = Pipeline(
    steps=[
        ("extract_features", raw_data_processor),
        ("scale", scaler),
        ("rf", selected_model),
    ]
)


# ## Write Model to S3

# In[50]:


key = "thoth/github-pr-ttm/model"
filename = "model.joblib"
s3_resource = boto3.resource(
    "s3",
    endpoint_url=s3_endpoint_url,
    aws_access_key_id=s3_access_key,
    aws_secret_access_key=s3_secret_key,
)

with tempfile.TemporaryFile() as fp:
    joblib.dump(pipe, fp)
    fp.seek(0)
    s3_obj = s3_resource.Object(s3_bucket, f"{key}/{filename}")
    s3_obj.put(Body=fp.read())


# In[51]:


## Sanity Check
buffer = BytesIO()
s3_object = s3_resource.Object(s3_bucket, f"{key}/{filename}")
s3_object.download_fileobj(buffer)
model = joblib.load(buffer)
model


# ### Prediction data from raw PR data

# In[52]:


# take raw pr data and predict ttm classes
preds = model.predict(pr_df.reindex(X_test.index))
print(classification_report(y_test, preds))


# ## Conclusion
# 
# In this notebook, we explored various vanilla classifiers, namely, Naive Bayes, SVM, Random Forests, and XGBoost. The Random Forest classifier was able to predict the classes with a weighted average f1 score of 0.67 an accuracy of 68% when trained using all the available features.
# 
# Even though all models outperform the baseline (random guess), we believe there is still some room for improvement. Since the target variable of the github PR dataset is an ordinal variable, an ordinal classifier could perform better than the models trained in this notebook. 
# 
# As the immediate next step, we will to deploy the best model from this notebook as an inference service using Seldon.
