# Get Started

The aim of the AI for continuous integration project is to build an open AIOps community involved in developing, integrating and operating AI tools for CI by leveraging the open data that has been made available by OpenShift, Kubernetes and others. Check out the [project overview](../README.md) for a detailed overview of this project.

## Try it out yourself

There are interactive and reproducible notebooks for this entire [project](https://github.com/aicoe-aiops/ocp-ci-analysis) available for anyone to start using on the public [JupyterHub](https://jupyterhub-opf-jupyterhub.apps.zero.massopen.cloud/hub/login) instance on the [Massachusetts Open Cloud](https://massopen.cloud/) (MOC) right now!

1. To get started, access [JupyterHub](https://jupyterhub-opf-jupyterhub.apps.zero.massopen.cloud/), select log in with `moc-sso` and sign in using your Google Account.
2. After signing in, on the spawner page, please select the `ocp-ci-analysis:latest` image in the JupyterHub Notebook Image section from the dropdown and select a `Medium` container size and hit `Start` to start your server.
3. Once your server has spawned, you should see a directory titled `ocp-ci-analysis-<current-timestamp>`. Browse through, run the various notebooks and start exploring this project.
4. To interact with the S3 bucket and access the stored datasets, make sure you have a `.env` file at the root of your repo. Check [.env-example](../.env-example) for an example `.env` file and open an [issue](https://github.com/aicoe-aiops/ocp-ci-analysis/issues) for access credentials.

You can find more information on the various notebooks and their purpose [here](content.md).

If you need more help navigating the Operate First environment, we have a few [short videos](https://www.youtube.com/playlist?list=PL8VBRDTElCWpneB4dBu4u1kHElZVWfAwW) to help you get started.
