# Build AIOps tools on Operate First Workshop


## Workshop Outline

Outline for the upcoming DevConf US 2022 and Scale 19x.
We would like to create a workshop flow with documentation such that attendees can follow along using this documentation during the workshop and even after the workshop.

Location: We plan to add a workshop chapter to the [existing Jupyterbook](https://aicoe-aiops.github.io/ocp-ci-analysis/README.html) and have a step-by-step tutorial for each of the following steps:

### Pre-requisites or  proposed Part 1 of Scale workshop:

1. Introduce the Operate First initiative
2. Introduce Open Data Hub
3. Introduce the services Jupyterhub, Openshift, Trino, Cloudbeaver, S3 storage, Superset, Kubeflow, Seldon.
4. High level data science model development process. What are the steps that are involved and how Open Data Hub supports that?
  - Feature engineering
  - Model training
  - Model deployment and monitoring
5. Onboarding (access and login) to all the tools and services. Hello world workloads?

### Part 2 Scale Workshop / DevConf.US Workshop

1. **Slides:**  Introduction to AIOps, project AI4CI and Operate First (refresher for folks who attended Part 1 and background for folks who are dropping into Part2).
2. **Introduction and Onboarding:** This section will consist of introduction to the workshop for the attendees and consist of onboarding and login details for the operate first environment (tools like Jupyterhub, Openshift console, Trino, Cloudbeaver, S3 bucket, Superset).
3. The assumption is that some sort of onboarding has already been processed in Part 1 of the workshop.
4. **Model Development:**
  - Spawn the image for this repository for Jupyterlab.
  - Introduce the 3 notebooks ( data collection, feature engineering (+ trino table creation maybe), model training)
  - Set up kubeflow runtime image and configurations
  - Run a pre-existing elyra pipeline
5.  **ML Pipeline:** This section will consist of Kubeflow UI and demonstrate how we automate the ML workflow from collecting the data, feat eng, exporting tables to trino)
6. **Model Deployment:** Deploy your trained model
  - Instructions on how to create an image with prediction script and requirements (only documentation, not being done live at workshop)
  - Create a deployment using Seldon operator.
  - Create a route to the Seldon service.
  - Interact with an already trained model from Jupyeterlab.
  - Section where users can directly access and interact with the service using the pre-trained model and existing URL or the newly created route to the model.
7. **SQL Query Engine (optional)**: Sending the data via Trino notebook, accessing and interacting with the dataset through Cloudbeaver UI.
8. **Visualization Dashboard (optional):**
  - Import a dataset using the table added on Trino.
  - Create charts and dashboard
