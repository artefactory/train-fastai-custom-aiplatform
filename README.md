# Train a FastAI model using a custom container hosted in GCP AI Platform
- [Introduction](#introduction)
- [Pre-requisites](#pre-requisites)
- [Set-up your environment](#set-up-your-environment)
- [Train the model](#train-the-model)


# Introduction
AI Platform is a fully-managed cost-effective service provided by Google Cloud Platform, allowing users to train, deploy and use ML models directly on cloud. 
We'll see in this repo how to setup AI Platform for ML models training, and how to automatically and easily train text-classifiers with FastAI.


# Pre-requisites
To follow this tutorial, be sure to have in possession the following elements:
- A GCP environment, with access to Cloud Container, Storage and AI Platform
- A VM (or your personal computer) with a GPU. We’ll consider here that you can access a Google VM with GPU enabled. You can create a ready-to-use Deep Learning VM from Google’s Market Place. It is not necessary to have a VM with a GPU, but it will allow you to test if everything is running OK before pushing it to AI Platform
- Docker and nvidia-docker installed on your VM. You can find how to install them here:
  - https://docs.docker.com/engine/install/ubuntu/ to install docker
  - https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker to install nvidia-docker
- A labelled dataset to train your model on (one is provided in this tutorial if you just want to know how to train the model), where the labels to predict are one-hot encoded (i.e there is a column for every label, that contains True/False or 0/1 for each texts).


# Set-up your environment
The first thing you’ll have to do will be to setup your working environment:
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
  - BUCKET_NAME: The name of your bucket in GCS (e.g "my_bucket")
  - PROJECT_ID: The name of your GCP Project, accessible by doing:
  > ```python
  > gcloud config list project --format "value(core.project)"
  > ```
  - REGION: The region you operate in, choose one with GPUs available (e.g "europe-west1" for Europe)
  - IMAGE_REPO_NAME: The name of the folder where will be stored your containers in Container Registry, (e.g "fastai_gpu_container")
  - IMAGE_TAG: The tag name you want to give to your future container (e.g "french_training")
  - IMAGE_URI: The URI to your future container:
  > ```python
  > export IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG
  > ```
  - MODEL_DIR: The directory in your bucket that will store your trained model (e.g "models")
  - JOB_NAME: AI Platform job name (e.g "fastai_gpu_french_job1")

- Create a bucket on GCS with the same architecture as in the following drive, and upload your files in it:
  https://drive.google.com/drive/folders/1JHfan6SFOXz5X0h49GOdORQQI3-rjpJX?usp=sharing.

  You will find on this Drive all the files that are necessary to train a classifier in 5 languages (English, French, Korean, Japanese or Chinese):
  - A labelled dataset to train your classifier (that can be replaced by your own)
  - A pretrained language model (forward or backward may be available) defined by three files:
    - A vocab.pkl file corresponding to the vocabulary of the LM
    - A config.json file where the configuration of the model is encoded
    - A weights.pth file containing the weights of the model
  You will be able to specify if you want to use a backward model to train your classifier if it's available. English LM being already handled by Fastai, no pretrained LM is necessary for this language.


# Train the model
## Copy the repo on your VM
> ```python
> $ git clone https://github.com/artefactory/train-fastai-custom-aiplatform.git
> $ cd train-fastai-custom-aiplatform
> ```


## Take a look at the configuration file and make necessary changes
The fastai_config.py file contains all the variables that will be useful to train your model, and the names of the files to fetch your model with. Here's the list of all the variables:
  - Information describing your dataframe (You need to make sure these correspond to your actual training labelled dataset):
    - TEXT_COL_NAME: Name of the text column in your labelled dataset
    - LABEL_LIST: List of the labels to be taken into account (referring to the columns of your one-hot encoded dataset)
    
  - Name of the variables that will be created during the training:
    - LABEL_COL_NAME: Name of the column containing the separated labels, that will be created from your one-hot encoded columns
    - LABEL_DELIM: Delimiter between your labels in your LABEL_COL_NAME column
    - OTHER_LABEL_NAME: Label to give when no label is assigned to a text
    - PREDICTION_COL_NAME: Name of the column that will store the predictions during inference on test dataset (to assess model's performances)
    
  - Parameters that will have an influence on your training
    - METRIC_TO_MONITOR: Metric to be monitored during the training to improve your model
    - RANDOM_STATE: Random state of train/test/valid split of dataframe
    - VAL_SIZE: Proportion of samples to be used for validation
    - TEST_SIZE: Proportion of samples to be used for testing
    - PREDICTION_THRESHOLD: Threshold from which the model will assign a prediction to a label for a given text
 
The other variables refer to the names and locations of all the files to be used during the training.


## Build image.
The image is built based on Nvidia Cuda 10.1-devel image, which should automatically be pulled when trying to build.
> ```python
> docker build -f Dockerfile -t $IMAGE_URI ./
> ```
You’ll get a success message if the image is built correctly, or an error message otherwise. Be really careful at the python, pip, pytorch and nvidia version you choose, because incompatibilities will cause the package you want to install to fail without any specific reason.
Build may fail due to a lack of RAM so be sure that the machine you're using can handle it. You shouldn't have any problem if you use Google's GPU-enabled VMs.


## (Optional) Run image before pushing to ensure its well-behavior
Once your image is built, you can run the container with basic parameters just to ensure everything works. Run this command and wait for it to complete before following this tutorial.
> ```python
> docker run --runtime=nvidia $IMAGE_URI --epochs 1 --bucket-name $BUCKET_NAME
> ```

Note: We only specified here a few parameters:
- --runtime=nvidia: We ask our container to use the GPU (if your machine has one obviously)
- --epochs 1: We train our model on only 1 epoch, to minimize the time needed to train (we don't aim performances, we only want to see if the training goes well)
- --bucket-name $BUCKET_NAME: We specifiy the name of the bucket to get the training files from


## Push your container
After running the image and ensured it worked, you can push it to GCR using the following command:
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
> --lang=fr \
> --bucket-name=$BUCKET_NAME \
> --model-dir=$MODEL_DIR \
> ```

We specified a few parameters here:
  - The name of the job
  - The type of scaling we want, BASIC_GPU here
  - The region we want our running machine to be
  - The URI of our custom container
  - Training arguments, that have been defined in the args_getter.py file
  
These training arguments are not all necessary if they have a default value, but allow you to customize the training of your model. They can be specified after the "-- \" when running your command:
  - --lang: The language of our dataset, to choose among "en", "fr", "ja", "ko" and "zh" (default = "fr")
  - --bucket-name: The name of your bucket on GCS
  - --model-dir: The name of the directory to store your trained model on GCS
  - --drop-mult: The value of the Drop-Multiplier to decrease risk of overfitting (default = 0.3)
  - --batch-size: The size of text batches to be processed at the same time during model's fitting (default = 16)
  - --epochs: The number of epochs to train your model during each cycle of fitting (default = 8)
  - --bw: The type of pretrained LM you want to use (forward or backward). Specify '--bw' if you want to use a backward model, nothing otherwise.


The training might take some time depending on the training arguments you chose (especially the number of epochs)


## Follow your model training

To follow the training of your model, you can go to AI Platform menu on GCP, select "Jobs", and click on the name of your job. You'll be able to view the logs, and some basic information about the job, like the input arguments file, or the % of use of the CPU/GPU.

You can also see the status of your training by running the command
> ```python
> gcloud ai-platform jobs describe $JOB_NAME
> ```

The evolution of the training itself will be displayed in the logs, where you'll be able to see the model fitting for each epoch.

Once the job is complete, your model will be available in your bucket, in the folder $MODEL_DIR, along with the performance metrics of your model for each label, stored in a JSON file, that you'll also be able to see in the logs.


# References
- [Getting started with AI Platform Training with custom containers](https://cloud.google.com/ai-platform/training/docs/custom-containers-training#submit_a_hyperparameter_tuning_job)
- [Use the GPU within a Docker container](https://blog.roboflow.com/use-the-gpu-in-docker/)
