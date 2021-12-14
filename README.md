# Ensembles

## What is this project?
This project contains an implementation of Random Forest and Gradient Boosting algorithms, visualized in a Flask web server.

## Installation
First you need to get the Docker container. There are to ways to do that:
1. Clone this repository and execute `build.sh` script to build the Docker container yourself:
    ```zsh
    scripts/build.sh
    ```
2. Pull the container from [dockerhub.com](https://dockerhub.com) and tag it.
    
    Choose `ml_server-amd64` or `ml_server-arm64` according to your architecture.
    ```zsh
    docker pull antonyfrolov/ensembles:ml_server-amd64
    docker tag antonyfrolov/ensembles:ml_server-amd64 ml_server
    ```
To run the Docker container execute `run.sh` script:
```zsh
scripts/run.sh 5000
```
Then connect to port 5000 ([click](http://127.0.0.1:5000/)).

If port 5000 is not available, you can change it by passing it as an argument to `run.sh`.
```zsh
scripts/run.sh <port>
```

**Make sure you have permissions to execute the scripts:**
```zsh
chmod +x scripts/build.sh scripts/run.sh 
```

## Application interface

### Model creation page

This page is a simple form for choosing the model and its parameters. (For Random Forest leave the **learning rate** field blank)

### Training page

Next page allows you to upload a training dataset and an optional validation dataset. You can also choose a fraction of your training dataset to be used as validation data by specifying a float number in a corresponding field.

The default name for target feature is `TARGET`, but you can specify another one in **Target feature** field.

Auto-preprocessing treats all float features as numeric, binary integer features as binary and all other features as categorical.
You can specify types of features yourself in the fields above. To do that provide lists of feature names separated by `', '`.

### Main page

On the main page you can find model parameters as well as names of training and validation datasets.

You can create a new model, train existing model again or proceed to evaluation page through the links.

To make a prediction first load a test dataset and click **Predict!** button. All columns of the test dataset should be exactly the same as the columns from the training one, except for the target column which must not be included. The prediction will be downloaded as a `.txt` file.

### Evaluation page

The evaluation page consists of two graphs:
1. Training and validation RMSE for each number of estimators.
2. Total training time for each number of estimators.

Also there you can find best train and validation RMSE values and total training time.
