# Pre-requisites

In this workshop we will learn how to build an AIOps tool from scratch on the [Operate First](https://www.operate-first.cloud/) community cloud and learn how to move your AI workloads to this cloud. We will introduce concepts like AIOps, Data Science Model development process, the project [opendatahub](https://opendatahub.io/) and share an end to end example of cloud native data science development process on the open Operate First community cloud.

Following are some pre-requisites for the workshop:

1. **Laptop with a working browser** - Most of our workloads will be on the Operate First community cloud and for access to the tools and services you would need a Laptop with a working browser (we recommend Google Chrome).

2. **Github Account** - Having a Github Account before the workshop is preferred but not compulsory. Refer to [this chapter](./git_setup.md) on Git setup instructions. We will be setting aside some time during the wokrshop to setup Github accounts for attendees who do not have accounts already.

3. **Familiarity with Python** - The AIOps tools has been built using the Python programming language. Familiarity with Python is preferred but not compulsory. One can still complete the workshop without prior Python knowledge.

You do not need to be familiar with these any of tools being used in this workshop, and you should have around an intermediate level of comfort with technology and learning new technical concepts.

### A note on the Environment Used

All of the services used in this work shop are deployed on [OpenShift Container Platform][ocp].
The entire suite of tools and their configurations can be found [here][configs].
Anyone with an Openshift Cluster (v4.9+) can leverage these configurations and recreate the Workshop Environment.

[ocp]: https://www.redhat.com/en/technologies/cloud-computing/openshift/container-platform
[configs]: https://github.com/operate-first/apps/tree/master/workshops/aiops-tools
