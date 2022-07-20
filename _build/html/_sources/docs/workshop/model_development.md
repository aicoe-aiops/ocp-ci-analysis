# Model Development


## Set Environment Variables

This is an important step to get right for successfully being able to run the notebooks.

* Go to [Jupyterhub](https://jupyterhub-aiops-tools-workshop.apps.smaug.na.operate-first.cloud/hub/spawn) and select the ocp ci analysis image and the “Workshop” container size
* Open the terminal and clone your forked Github repository in Jupyterhub.

`git clone https://github.com/{YOUR-USERNAME}/ocp-ci-analysis`

* Set up env variables at the root of the repository.

`cd ocp-ci-analysis`

`vi .env`

Add the following contents to the .env file. For Bucket and Trino credentials refer to this link.


Save the file and start running the notebooks.

**Note**: We would be downloading Pull Requests data from a Github repository.


For faster download, choose a repository which does not have too many PRs. For example purposes, we have chosen a repository with ~1000 PRs

```
    GITHUB_REPO= Github repository that you wish to download
    GITHUB_ORG= Github organization that the repository belongs to. If it's on your personal account, this will be your username.
    S3_ACCESS_KEY= S3 bucket access key
    S3_ENDPOINT_URL= S3 bucket endpoint
    S3_BUCKET= S3 bucket name
    S3_SECRET_KEY= S3 bucket secret key
    CEPH_BUCKET= S3 bucket name
    CEPH_BUCKET_PREFIX= set this to your username, this is the location/key where the files will be stored on S3 storage
    CEPH_KEY_ID= S3 bucket access key ID
    CEPH_SECRET_KEY= S3 bucket secret key
    GITHUB_ACCESS_TOKEN_TODAY= Your Github personal access token generated from the previous step
    TRINO_USER= trino user
    TRINO_PASSWD= trino password
    TRINO_HOST= trino host
    TRINO_PORT= trino port
    CHOSEN_MODEL= Model that you wish to choose for deployment. Either one of 'rf', 'xgbc', 'svc', 'gnb'

```

## Model Development

In this workshop, we will be training a machine learning model using tools and services available on Operate First.

The aim of this model is to take a Github repository of interest and predict the time that it will take to merge a new Pull Request. For that purpose, we frame the “time taken to merge a PR” as a classification problem where we predict whether the time taken to merge a PR falls within one of a few predefined time ranges.

Browse through the following notebooks to understand the model development process,    to directly see the model development running in a pipeline or in action, skip to the next section

### Data collection

In order to collect the data from Github repositories, we use the thoth-station [MI- Scheduler](https://github.com/thoth-station/mi-scheduler) that collects and analyzes metadata information from Github repositories and stores them on ceph object storage. We use the MI-Scheduler tool to collect Pull Request data from a repository of your choice.

### Feature engineering

After collecting the data, we perform some initial exploration such as correlation analysis on the dataset to discover any interesting insights. We then engineer features which are needed to train a classification model which predicts the time to merge of a PR.


We transform the input columns obtained from pull requests such as size of a PR, types of files added in a PR, description of a PR into various features which can be ingested by an ML Model.

### Model training

After performing initial data analysis and feature engineering, we train a machine learning model to classify the time_to_merge values for PRs into one of 10 bins (or "classes"). To train this model, we use the features engineered from the raw PR data. We explored various vanilla classifiers, like Naive Bayes, SVM, Random Forests, and XGBoost and save the best performing model on S3 storage.
