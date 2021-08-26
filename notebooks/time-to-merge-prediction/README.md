# Time to Merge Prediction

The aim of this ML Problem is to take a Github repository of interest and predict the time that it will take to merge a new Pull Request. For that purpose, we frame the “time taken to merge a PR” as a classification problem where we predict whether the time taken to merge a PR falls within one of a few predefined time ranges.

## Github Dataset

In order to collect the data from Github repositories, we use the thoth-station [MI- Scheduler](https://github.com/thoth-station/mi-scheduler) that collects and analyzes metadata information from Github repositories and stores them on ceph object storage. We use the MI-Scheduler tool to collect Pull Request data from the [OpenShift origin](https://github.com/openshift/origin) repository.

## Feature Engineering

After collecting the data, we perform some initial exploration such as correlation analysis on the dataset to discover any interesting insights. We then engineer features which are needed to train a classification model which predicts the time to merge of a PR.

We transform the input columns obtained from pull requests such as size of a PR, types of files added in a PR, description of a PR into various features which can be ingested by an ML Model.

   * [Feature Engineering Notebook](../notebooks/data-sources/oc-github-repo/github_PR_EDA.ipynb)

## Model Training

After performing initial data analysis and feature engineering, we train a machine learning model to classify the time_to_merge values for PRs into one of 10 bins (or "classes").

To train this model, we use the features engineered from the raw PR data. We explored various vanilla classifiers, like Naive Bayes, SVM, Random Forests, and XGBoost.

   * [Model Training Notebook](time_to_merge_model.ipynb)


## Model Deployment

To make the machine learning model available at an interactive endpoint,  we serve the model yielding the best results into a Seldon service. We create an sklearn pipeline consisting of 2 steps, scaling of the input features and the classifier itself.

   * Saved sklearn model pipeline on `opf-datacatalog` bucket: `github/ttm-model/pipeline/model.joblib`
   * Interactive model endpoint: http://ttm-pipeline-opf-seldon.apps.zero.massopen.cloud/predict
   * [Model Inference Notebook](model_inference.ipynb)

This service once integrated with a Github repo, can provide newly submitted PRs with a time to merge estimate.
