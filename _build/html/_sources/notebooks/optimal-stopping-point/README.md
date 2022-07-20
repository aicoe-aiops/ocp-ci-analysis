# Optimal Stopping Point Prediction

The aim of this ML problem is to predict an optimal stopping point for CI tests based on their test duration (runtimes). We perform an initial data analysis as well as feature engineering on the testgrid data. Furthermore, we also calculated the optimal stopping point by identifying the distribution of the test duration values for different CI tests and comparing the distributions between the passing and failing tests.

## Dataset

 In order to achieve the optimal stopping point, we would be looking into the  [testgrid](https://testgrid.k8s.io/) data for all the passing and failed tests and find the distribution type for the `test_duration` metric. The `test_duration` metric tracks the time it took for a test to complete its execution. We can visualize the distribution of the `test_duration` metric across various testgrid dashboards and jobs. Based on the distribution type identified, we can find a point after which the test has a higher probability of failing.

## Feature Engineering

After fetching the data, we approximate the distributions of the `test_duration` metric and also check its goodness of fit for different TestGrid tests across all TestGrid dashboards and grids. Based on the type of distribution identified, we can calculate the probability of the test failing.

We fetch data for all the passing and failed tests and find the distribution type for the `test_duration` metric. The `test_duration` metric tracks the time it took for a test to complete its execution. We can visualize the distribution of the `test_duration` metric across various testgrid dashboards and jobs. Based on the distribution type identified, we can find top two distributions based on betterment of fit. Probability density plots are used to understand data distribution for a continuous variable and likelihood (or probability) of obtaining a range of values that the continuous variable can assume. The area under the curve contains the probabilities for the test duration values. In a test_duration probability distribution function, the area under the curve from 0 to a given value represents the probability that the test_duration is less than or equal to that value.

   * [Probability To Fail notebook](../data-sources/TestGrid/metrics/probability_to_fail.ipynb)

## Model Training

After performing initial data analysis and calulating the probability to fail, we predict the optimal stopping for a given test based on their test duration(runtimes). We find the best distribution(s) for the given test, and find an optimal stopping point for the test by finding the point where the probability of the test failing is greater than the probabilty of the test passing.

   * [Model Training Notebook](osp_model.ipynb)

## Model Deployment

To make the machine learning model available at an interactive endpoint,  we serve the model yielding the best results into a Seldon service.

   * Interactive model endpoint: http://optimal-stopping-point-ds-ml-workflows-ws.apps.smaug.na.operate-first.cloud/predict
   * [Model Inference Notebook](model_inference.ipynb)
