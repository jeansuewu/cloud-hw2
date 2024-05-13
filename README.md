# Clouds

This repository contains code and instructions for deploying a cloud classification model into production. The model classifies clouds into one of two types based on features generated from cloud images.

## Assignment Overview

The assignment involves moving code from a Jupyter notebook (`clouds.ipynb`) into modular functions within Python modules and scripts. These functions are composed into a single `pipeline.py` script to ensure reproducibility across environments. 

Key steps include:
- Pulling out configurations into a YAML file.
- Moving code into modular functions.
- Adding unit tests for the functions.
- Writing a Dockerfile for running the pipeline.
- Ensuring compliance with PEP8 and passing pylint linting.
- Setting the S3 bucket for uploading pipeline artifacts.

## Setup Instructions

To set up the project, follow these steps:

1. Clone the repository:

   ```bash
   git clone <repository-url>
    ```
2. Set up the configuration file (`config.yaml`) with necessary parameters.

3. Configure AWS credentials:
- Install and configure the AWS Command Line Interface (CLI).
- Ensure that your IAM user has permissions to access the S3 bucket where you will upload pipeline artifacts. You can configure these permissions in the IAM console.

## Build the Docker image

To build the Docker image, run the following command in the project directory:

```bash
docker build -t cloud-classifier .
```

## Run the entire model pipeline

To run the Docker container, execute the following command:

```bash
docker run cloud-classifier
```

To run the Docker container with your AWS profile and mount your project directory, use the following command:

```bash
docker run -e AWS_PROFILE=personal -v ~/.aws:/root/.aws:ro -v $(pwd):/pipeline cloud-classifier
```
This command does the following:

- Sets the AWS_PROFILE environment variable to "personal" inside the container, allowing authentication with your AWS credentials.

- Mounts the ~/.aws directory from your local machine to /root/.aws inside the container in read-only mode (ro), ensuring that your AWS credentials are accessible inside the container.

- Mounts the current directory ($(pwd)) from your local machine to /pipeline inside the container, providing access to your project files and allowing the container to execute the necessary commands.

## Run unit tests
Start an interactive shell inside the Docker container:
```bash
docker run -it --rm cloud-classifier /bin/bash
```
Once inside the container, run the following command to execute the tests:
```bash
pytest tests
```
