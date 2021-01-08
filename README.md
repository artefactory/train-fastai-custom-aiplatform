# Train a FastAI model using a custom container hosted in GCP AI Platform
- [Introduction](#introduction)
- [Pre-requisites](#pre-requisites)
- [Set-up your environment](#set-up-your-environment)
- [Train the model](#train-the-model)

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


# Train the model
## Copy the repo on your VM
> ```python
> git clone https://github.com/artefactory/train-fastai-custom-aiplatform.git
> ```

## Build image.
The image is built based on Nvidia Cuda 10.1-devel image, which should automatically be pulled when trying to build.
> ```python
> docker build -f Dockerfile -t $IMAGE_URI ./
> ```
You’ll get a success message if the image is built correctly, or an error message otherwise. Be really careful at the python, pip, pytorch and nvidia version you choose, because incompatibilities will cause the package you want to install to fail without any specific reason


## Run image to see everything is ok
Once your image is built, you can run the container to see if everything works (especially cuda)
Permission to upload to GCS may be denied (403), but should work when doing it with AIP, so no worries if it's the only thing that causes errors
> ```python
> docker run --runtime=nvidia $IMAGE_URI --epochs 2 --bucket-name $BUCKET_NAME
> ```
Don't forget to specify that you want to run on GPU (nvidia runtime), and the number of epochs to train your model on

## Push your container
When you made sure that everything worked, you can push your image to GCR using the following command:
> ```python
> docker push $IMAGE_URI
> ```
The container will be visible in Container registry.


## Run AI Platform job
Now that your custom container is in GCR, everything is ready. You can start the training on AI Platform using a simple command, from any machine you want, assuming that it has access to your GCP project and to AI Platform.

You just have to run the following command for everything to start running. You’ll receive a notification when the training is ready

> ```python
> gcloud ai-platform jobs submit training $JOB_NAME \
> --scale-tier BASIC_GPU \
> --region $REGION \
> --master-image-uri $IMAGE_URI \
> -- \
> --epochs=8 \
> --bucket-name=$BUCKET_NAME \
> --model-dir=$MODEL_DIR
> ```
We specified a few parameters here:
  - The name of the job
  - The type of scaling we want, BASIC_GPU here
  - The region we want our running machine to be
  - The URI of our custom container
  - Our arguments:
    - The number of epochs
    - The name of our bucket in GCS
    - The name of the directory to store our trained model
This might take some time depending on how many epochs you chose to train your model on

- You can view the status of your job with the command
> ```python
> gcloud ai-platform jobs describe $JOB_NAME
> ```

Once the job is complete, your model will be available in your bucket, in the folder $MODEL_DIR
