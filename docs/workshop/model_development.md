# Model Development

In this section, we will learn how to train an AIOps model from scratch. The model that we plan to train in particular is a Time to Merge prediction model which offers time estimates on when a Pull Request on a Git repository will be merged. To train such a model we look at historic pull request data of the repository (can also be extended to organization) and engineer features from the Pull Requests which can impact the time to merge of a Pull Request.

To train such a model, we will be using a Jupyterlab environment and make use of jupyter notebooks. The following steps will walk you through how to train a Github time to merge model.

## Set Environment Variables

This is an important step to get right for successfully being able to run the jupyter notebooks.

* Go to [Jupyterhub](https://jupyterhub-aiops-tools-workshop.apps.smaug.na.operate-first.cloud/hub/spawn) and select the `Openshift CI Analysis Notebook Image` and the “Workshop” container size.

If you do not have access to the Jupyterlab instance being used for this workshop, move to chapter on [Onboarding](./onboarding.md).

* Open the terminal and clone your forked Github repository in Jupyterhub.

First set up your github configuration in Jupyterhub

```
git config --global user.email "you@example.com"
git config --global user.name "Your Name"
```

Optional: You can also [generate an SSH key](https://docs.github.com/en/authentication/managing-commit-signature-verification) and add it to your github account at https://github.com/settings/keys.

`git clone https://github.com/{YOUR-USERNAME}/ocp-ci-analysis`

If you do not have a Github account or have not forked the repository, move to chapter on [Set up Git environment](./git_setup.md).

* Set up environment variables at the root of the repository.

Once you have cloned the repository, [cd](https://linuxize.com/post/linux-cd-command/) into the repository and create a `.env` file.

`cd ocp-ci-analysis`

`vi .env`

[Add](https://www.cs.colostate.edu/helpdocs/vi.html) the following contents to the .env file. For env file refer to this [link](https://vault.bitwarden.com/#/send/zTA4PuNJwEW6kq7ZAUnY8g/pf51QZhZcEQ4QCEN7Lbszw). The password to this vault will be shared during the workshop. Copy the contents of the file from bitwarden and paste it in your `.env` file.

For an example `.env` file refer to this [sample env file](../../notebooks/time-to-merge-prediction/workshop/env_example_workshop).

[Save](https://www.cs.colostate.edu/helpdocs/vi.html) the file and start going over the notebooks.

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
    GITHUB_ACCESS_TOKEN= Your Github personal access token generated from the previous step
    TRINO_USER= trino user
    TRINO_PASSWD= trino password
    TRINO_HOST= trino host
    TRINO_PORT= trino port
    CHOSEN_MODEL= Model that you wish to choose for deployment. Either one of 'rf', 'xgbc', 'svc', 'gnb'
    REMOTE=1

```

## Model Development

In this workshop, we will be training a machine learning model using tools and services available on Operate First.

To directly see the model development running in a pipeline or in action, skip to the [next section](./ml_pipeline.md)

This section comprises of 3 notebooks:
* [01_data_collection](../../notebooks/time-to-merge-prediction/workshop/01_data_collection.ipynb)
* [02_feature_engineering](../../notebooks/time-to-merge-prediction/workshop/02_feature_engineering.ipynb)
* [03_model_training](../../notebooks/time-to-merge-prediction/workshop/03_model_training.ipynb)

The aim of the model that we are training is to take a Github repository of interest and predict the time that it will take to merge a new Pull Request. For that purpose, we frame the “time taken to merge a PR” as a classification problem where we predict whether the time taken to merge a PR falls within one of a few predefined time ranges.

### Data collection

In order to collect the data from Github repositories, we use the thoth-station [MI- Scheduler](https://github.com/thoth-station/mi-scheduler) that collects and analyzes metadata information from Github repositories and stores them on ceph object storage. We use the MI-Scheduler tool to collect Pull Request data from a repository of your choice.

### Feature engineering

After collecting the data, we perform some initial exploration such as correlation analysis on the dataset to discover any interesting insights. We then engineer features which are needed to train a classification model which predicts the time to merge of a PR.

We transform the input columns obtained from pull requests such as size of a PR, types of files added in a PR, description of a PR into various features which can be ingested by an ML Model.

### Model training

After performing initial data analysis and feature engineering, we train a machine learning model to classify the time_to_merge values for PRs into one of 10 bins (or "classes"). To train this model, we use the features engineered from the raw PR data. We explored various vanilla classifiers, like Naive Bayes, SVM, Random Forests, and XGBoost and save the best performing model on S3 storage.
