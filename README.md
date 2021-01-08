# Title of the repo
Train a Fastai model using a custom container on AI Platform

# Introduction
Explanation of AI Platform

# Pre-requisites
To follow this tutorial, be sure to have in possession the following elements:
A GCP environment, with access to Cloud Container, Storage and AI Platform
A VM (or your personal computer) with a GPU. We’ll consider here that you can access a Google VM with GPU enabled
You can create a ready-to-use Deep Learning VM from Google’s Market Place
It is not necessary to have a VM with a GPU, but it will allow you to test if everything is running OK before pushing it to AI Platform
Some python code that trains a ML model, along its dataset. 
I personally used fastai 2.1.8 framework to train a text classifier using ULM FiT method
Docker and nvidia-docker installed on your VM 
https://docs.docker.com/engine/install/ubuntu/ to install docker
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker to install nvidia-docker

In order to create our custom container that will be used by AI Platform, we will create two separate files:
A training.py file that will download your labelled dataset from GCS, train your model and upload it to GCS
A Dockerfile to create a custom container to handle the training from your training.py file. The built container will be uploaded to GCR, and called from AI Platform
(optional) A config.yaml configuration file 

# Set-up your environment
The first thing you’ll have to do will be to setup your working environment. Here is a checklist of what to do:
Install and initiate Google Cloud SDK https://cloud.google.com/sdk/docs/install#linux
Allow yourself to run Docker without using sudo
sudo usermod -a -G docker ${USER}
Configure docker with gcloud
gcloud auth configure-docker
Create a bucket on GCS and upload your files in it

- Define variables
  - BUCKET_NAME: The name of your bucket
  - PROJECT_ID: The name of your GCP Project
  - REGION: The region you operate in (choose one with GPUs available, for Europe its europe-west1 usually)
  - IMAGE_REPO_NAME: The name of the folder where will be stored your containers in Container Registry
  - IMAGE_TAG: The tag name of your future container
  - IMAGE_URI: The URI to your future container, defined by:
    IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG
  - MODEL_DIR: The directory in your bucket that will store your trained model
  - JOB_NAME: AI Platform job name


# TBC
- Create Dockerfile and training.py file

- Build image

- Run image to see if everything works

- Push image

- Run AI Platform job

