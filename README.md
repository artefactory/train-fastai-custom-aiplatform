# Train a FastAI model using a custom container hosted in GCP AI Platform
- [Introduction](#introduction)

# Introduction
Explanation of AI Platform and what we're going to do

# Pre-requisites
To follow this tutorial, be sure to have in possession the following elements:
- A GCP environment, with access to Cloud Container, Storage and AI Platform
- A VM (or your personal computer) with a GPU. We’ll consider here that you can access a Google VM with GPU enabled. You can create a ready-to-use Deep Learning VM from Google’s Market Place. It is not necessary to have a VM with a GPU, but it will allow you to test if everything is running OK before pushing it to AI Platform
- Docker and nvidia-docker installed on your VM. You can find how to install them here:
  - https://docs.docker.com/engine/install/ubuntu/ to install docker
  - https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker to install nvidia-docker

# Set-up your environment
The first thing you’ll have to do will be to setup your working environment. Here is a checklist of what to do:
- Install and initiate Google Cloud SDK https://cloud.google.com/sdk/docs/install#linux
- Allow yourself to run Docker without using sudo
> ```python
> sudo usermod -a -G docker ${USER}
> ```
- Configure docker with gcloud
> ```python
> gcloud auth configure-docker
> ```

- Define global variables
  - BUCKET_NAME: The name of your bucket in GCS
  - PROJECT_ID: The name of your GCP Project, accessible by doing:
> ```python
> gcloud config list project --format "value(core.project)"
> ```
  - REGION: The region you operate in (choose one with GPUs available, for Europe its europe-west1 usually)
  - IMAGE_REPO_NAME: The name of the folder where will be stored your containers in Container Registry, I chose "fastai_gpu_container"
  - IMAGE_TAG: The tag name you want to give to your future container
  - IMAGE_URI: The URI to your future container, defined by:
> ```python
> export IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG
> ```
  - MODEL_DIR: The directory in your bucket that will store your trained model (e.g "models")
  - JOB_NAME: AI Platform job name

- Create a bucket on GCS and upload your files in it, following the same architecture than in this drive:
  https://drive.google.com/drive/folders/1JHfan6SFOXz5X0h49GOdORQQI3-rjpJX?usp=sharing

# TBC
- Copy the repo on your VM

- Build image
> ```python
> docker build -f Dockerfile -t $IMAGE_URI ./
> ```

- Run image to see everything is ok
> ```python
> docker run --runtime=nvidia $IMAGE_URI --epochs 2 --bucket-name $BUCKET_NAME
> ```

- Push image
> ```python
> docker push $IMAGE_URI
> ```


- Run AI Platform job
> ```python
> gcloud ai-platform jobs submit training $JOB_NAME \
> --scale-tier BASIC_GPU \
> --region $REGION \
> --master-image-uri $IMAGE_URI \
> -- \
> --epochs=5 \
> --bucket-name=$BUCKET_NAME \
> --model-dir=$MODEL_DIR
> ```
