# Train a FastAI model using a custom container hosted in GCP AI Platform
- [Introduction](#introduction)
- [Pre-requisites](#pre-requisites)
- [Set-up your environment](#set-up-your-environment)
- [Train the model](#train-the-model)


# Introduction
Ever wanted to quickly train a text classifier without having to worry about "which framework should I use ?" or "should I use a VM to run on GPU?" ? Well, this repo should help you.

Thanks to AI Platform, a fully-managed cost-effective service provided by Google Cloud Platform, you can now train ML models directly on cloud using any framework you want by deploying a custom Docker container hosting your training code on GCR, and calling it from your machine with a simple command. 

You will find in this repo everything you need to easily build and deploy a custom Docker container to train a text-classifier on AI Platform with FastAI, a PyTorch wrapper that allows you to train powerful models with only few samples thanks to transfer learning. Indeed, FastAI text classifiers are created using ULM FiT method, which allows you to create classifiers using pre-trained Language models (you can find more details [here](https://towardsdatascience.com/understanding-language-modelling-nlp-part-1-ulmfit-b557a63a672b#:~:text=Universal%20Language%20Model%20FIne%2DTuning,by%20XLNet%20in%20text%20classification%5D.), but don't worry, you can follow this tutorial without knowing anything about language models).

By simply modifying few lines of code, you'll be able to create a text classifier adapted to your particular use case, whether you need to assign sentiment scores to movie reviews of a french website, or predict if a Japanese Instagram publication is about sport or food. 


# Pre-requisites
To follow this tutorial, be sure to possess the following elements:
- A GCP environment, with access to Container Registry, Cloud Storage and AI Platform
- A labelled dataset to train your model on, where the labels to predict are one-hot encoded (i.e there is a column for every label, that contains True/False or 0/1 for each texts).
For example, we'll use in this tutorial a dataset that has three columns : text_clean | categ1 | categ2, where "text_clean" contains the text to train the model on, and where "categ1" and "categ2" are filled with 0 or 1 to indicates if a text belongs to one of these categories. A sample is available in "sample" folder.
- A pre-trained universal language model adapted to your target language if you're working on another language than English. One for French, Korean, Japanese and Chinese is available on the Google Drive (link below in the ReadMe). This pre-trained LM is composed of a weights.pth file, a vocab.pkl file and a spm.model file (the last one is used by the sentencepiece tokenizer)
- (Optional) A VM or your personal computer with a GPU and enough RAM to build the Docker Image, which is not necessary but will allow you to test if everything is running OK before pushing your container to AI Platform. 
We’ll consider here that you can access a Google VM with GPU enabled, which can be easily created using a ready-to-use Deep Learning VM from Google’s Market Place.

# Set-up your environment
The first thing you’ll have to do will be to setup your working environment:
- Install Docker and nvidia-docker on your VM. You can find how to install them here:
  - https://docs.docker.com/engine/install/ubuntu/ to install docker
  - https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker to install nvidia-docker
- Install and initiate Google Cloud SDK https://cloud.google.com/sdk/docs/install#linux
- Allow yourself to run Docker without using sudo
> ```python
> sudo usermod -a -G docker ${USER}
> ```
- Configure docker with gcloud
> ```python
> gcloud auth configure-docker
> ```

- Create a bucket on GCS with the same architecture as in the following drive, and upload your files in it:
  https://drive.google.com/drive/folders/1JHfan6SFOXz5X0h49GOdORQQI3-rjpJX?usp=sharing.

  You will find on this Drive all the files that are necessary to train a classifier in 5 languages (English, French, Korean, Japanese or Chinese):
  - A labelled dataset to train your classifier (that can be replaced by your own)
  - A pretrained language model (forward or backward may be available) defined by four files:
    - A vocab.pkl file corresponding to the vocabulary of the LM
    - A config.json file where the configuration of the model is encoded
    - A weights.pth file containing the weights of the model
    - A spm.model file corresponding to the sentencepiece tokenizer used by the model
  You will be able to specify if you want to use a backward model to train your classifier if it's available (which is usually more efficient). English LM being already handled by Fastai, no pretrained LM is necessary for this language.

If your architecture in your bucket is different than the Drive's, you can modify the scheme of your files path in trainer/fastai_config.py

- Define global variables
  - BUCKET_NAME: The name of your bucket in GCS (e.g "my_bucket")
  - PROJECT_ID: The name of your GCP Project, accessible by doing:
  > ```python
  > gcloud config list project --format "value(core.project)"
  > ```
  - REGION: The region the training will operate in. Be sure to choose one with GPUs available (e.g "europe-west1" for Europe)
  - IMAGE_REPO_NAME: The name of the folder where your containers will be stored in Container Registry, (e.g "fastai_gpu_container")
  - IMAGE_TAG: The tag name you want to give to your future container (e.g "french_training")
  - IMAGE_URI: The URI to your future container:
  > ```python
  > export IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG
  > ```
  - MODEL_DIR: The directory in your bucket that will store your trained model (e.g "models")
  - JOB_NAME: An AI Platform job name (e.g "fastai_gpu_french_job1")


# Train the model
## Copy the repo on your VM
> ```python
> $ git clone https://github.com/artefactory/train-fastai-custom-aiplatform.git
> $ cd train-fastai-custom-aiplatform
> ```


## Take a look at the configuration file and make necessary changes
The fastai_config.py file contains variables that will be useful to train your model, and the names of the files to fetch your model with. 

Here's the list of the training parameters variables:
  - LABEL_COL_NAME: Name of the column containing the separated labels, that will be created from your one-hot encoded columns
  - PREDICTION_COL_NAME: Name of the column that will store the predictions during inference on test dataset (to assess model's performances)
  - RANDOM_STATE: Random state of train/test/valid split of dataframe
  - VAL_SIZE: Proportion of samples to be used for validation
  - TEST_SIZE: Proportion of samples to be used for testing
 
The other variables refer to the names and locations of all the files to be used during the training. If you use the same architecture and same file names as in the provided Google Drive, you shouldn't have to modify anything regarding that part.


## Build image.
The image is built based on Nvidia Cuda 10.2-devel image, which should automatically be pulled when trying to build.
> ```python
> docker build -f Dockerfile -t $IMAGE_URI ./
> ```
You’ll get a success message if the image is built correctly, or an error message otherwise. Be really careful of the python, pip, pytorch and nvidia version you choose, because incompatibilities will cause the package you want to install to fail without any specific reason. There shouldn't be any problem regarding that part though.
Build also may fail due to a lack of RAM so be sure that the machine you're using can handle it. You shouldn't have any problem if you use Google's GPU-enabled VMs.


## (Optional) Run the code in the container before pushing to ensure its well-behavior
Once your image is built, you can run the code in the container with basic parameters just to ensure everything works. Run this command and wait for it to complete before following this tutorial.
> ```python
> docker run --runtime=nvidia $IMAGE_URI --epochs 1 --bucket-name $BUCKET_NAME
> ```

Note: We only specified here a few parameters:
- --runtime=nvidia: We ask our container to use the GPU (if your machine has one obviously)
- --epochs 1: We train our model on only 1 epoch, to minimize the time needed to train (we don't aim performances, we only want to see if the training goes well)
- --bucket-name $BUCKET_NAME: We specifiy the name of the bucket to get the training files from


## Push your image to GCR
After running the image and ensured it worked, you can push it to GCR using the following command:
> ```python
> docker push $IMAGE_URI
> ```


## Run AI Platform job
Now that your custom container is in GCR, everything is ready to go. You can start the training on AI Platform using a simple command, from any machine you want, assuming that it has access to your GCP project and to AI Platform.
You just have to run the following command for everything to start running:

> ```python
> gcloud ai-platform jobs submit training $JOB_NAME \
> --scale-tier BASIC_GPU \
> --region $REGION \
> --master-image-uri $IMAGE_URI \
> -- \
> --lang=fr \
> --epochs=10 \
> --bucket-name=$BUCKET_NAME \
> --model-dir=$MODEL_DIR
> ```

We specified a few parameters here:
  - The name of the job
  - The type of scaling you want, BASIC_GPU here
  - The region you want your running machine to be
  - The URI of your custom container stored in GCR, that contains the training code
  - Training arguments, that have been defined in the args_getter.py file
  
These training arguments are not all necessary if they have a default value, but allow you to customize the training of your model. They can be specified after the "-- \" when running your command:
  - --bucket-name: The name of your bucket on GCS
  - --model-dir: The name of the directory to store your trained model on GCS
  - --model-filename: The name given to the trained model in GCS (default = "fastai_model.pth")
  - --score-filename: The name given to the json file containing your model performances in GCS (default = "label_scores.json")
  - --dataset-filename: The name of your labelled dataset to be retrieved in GCS (default = "labelled_dataset.csv")
  - --lang: The language of your dataset, to choose among "en", "fr", "ja", "ko" and "zh" (default = "fr")
  - --drop-mult: The value of the Drop-Multiplier to decrease risk of overfitting (default = 0.3)
  - --batch-size: The size of text batches to be processed at the same time during model's fitting (default = 16)
  - --epochs: The number of epochs to train your model during each cycle of fitting (default = 8)
  - --bw: The type of pretrained LM you want to use (forward or backward). Specify '--bw' if you want to use a backward model, nothing otherwise (you can see in the GCS bucket if a backward LM is available for the language you want to work on)
  - --monitored-metric: The metric to monitor to keep the best model during the training (default = "valid_loss")
  - --prediction-threshold: The threshold from which we consider a prediction probability is positive (default = 0.3)
  - --text-col: The name of the column containing the texts in your labelled dataset (default = "text")
  - --label-delim: The delimiter to be used when separating the various labels (default = ",")
  - --label-list: List of the labels in your training dataset, separated by label_delim (default = "categ1,categ2")
  - --other-label: The name of the label to be assigned when no other labeled has been assigned (default = "other")


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
