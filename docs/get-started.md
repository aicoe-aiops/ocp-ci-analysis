# Get Started

The aim of the AI for continuous integration project is to build an open AIOps community involved in developing, integrating and operating AI tools for CI by leveraging the open data that has been made available by OpenShift, Kubernetes and others. Check out the [project overview](../README.md) for a detailed overview of this project.

## Leverage Open Data Sources:

One of the goals of the AI4CI project is to build an open AIOps community centered around open datasets. As a part of this project, we interact with various open source CI data sources and make them available for others to leverage. A detailed overview of our efforts with the various data sources we are working with, along with data collection scripts, and exploratory analyses is available [here](content.md#data-engineering-metrics-and-kpis-for-ci).

## Try out the Notebooks:

There are interactive and reproducible notebooks for this entire [project](https://github.com/aicoe-aiops/ocp-ci-analysis) available for anyone to start using on the public [JupyterHub](https://jupyterhub-opf-jupyterhub.apps.zero.massopen.cloud/hub/login) instance on the [Massachusetts Open Cloud](https://massopen.cloud/) (MOC) right now!

1. To get started, access [JupyterHub](https://jupyterhub-opf-jupyterhub.apps.zero.massopen.cloud/), select log in with `moc-sso` and sign in using your Google Account.
2. After signing in, on the spawner page, please select the `ocp-ci-analysis:latest` image in the JupyterHub Notebook Image section from the dropdown and select a `Medium` container size and hit `Start` to start your server.
3. Once your server has spawned, you should see a directory titled `ocp-ci-analysis-<current-timestamp>`. Browse through, run the various notebooks and start exploring this project.
4. To interact with the S3 bucket and access the stored datasets, make sure you have a `.env` file at the root of your repo. Check [.env-example](../.env-example) for an example `.env` file and open an [issue](https://github.com/aicoe-aiops/ocp-ci-analysis/issues) for access credentials.

You can find more information on the various notebooks and their purpose [here](content.md).

If you need more help navigating the Operate First environment, we have a few [short videos](https://www.youtube.com/playlist?list=PL8VBRDTElCWpneB4dBu4u1kHElZVWfAwW) to help you get started.

## Interact with Dashboards:

As a part of AI4CI, we collect the relevant metrics and key performance indicators (KPIs) and visualize them using dashboards. You can view and interact with the publicly available dashboard [here](https://superset.apps.devconfus2021.aws.operate-first.cloud/superset/dashboard/ocp-ci-kpi-dashboard/).

## Interact with Model Endpoints:

* **Github Time to Merge Model**: We have an interactive endpoint available for a model which can predict the time taken to merge a PR and classifies it into one of a few predefined time ranges. To interact with the model, check out this [Model Inference Notebook](../notebooks/time-to-merge-prediction/model_inference.ipynb)

## Automate your AI/ML Workflows:

We use Elyra and Kubeflow pipelines to automate the various steps in the project responsible for data collection, metric calculation, ML analysis. To automate your notebook workflows you can follow this [guide](automating-using-elyra.md) or [tutorial video](https://youtu.be/bh5WpKq3W7Y).

## Video Playlist:

[Here](https://www.youtube.com/playlist?list=PL8VBRDTElCWoGwMhCp04rQFMcIhshv33U) is a video playlist for the AI4CI project which goes over different analyses and walks through various notebooks within the project.
