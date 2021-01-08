# Initializing image
FROM nvidia/cuda:10.1-devel

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    build-essential

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    python3-setuptools \
    python3-pip

RUN pip3 install pip==20.3.1

WORKDIR /root

# Install requirements
RUN pip3 install -r requirements.txt

# Installs cloudml-hypertune for hyperparameter tuning.
# It’s not needed if you don’t want to do hyperparameter tuning.
RUN pip3 install cloudml-hypertune

# Installs google cloud sdk, this is mostly for using gsutil to export model.
RUN wget -nv \
    https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
    mkdir /root/tools && \
    tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
    rm google-cloud-sdk.tar.gz && \
    /root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
        --path-update=false --bash-completion=false \
        --disable-installation-options && \
    rm -rf /root/.config/* && \
    ln -s /root/.config /config && \
    # Remove the backup directory that gcloud creates
    rm -rf /root/tools/google-cloud-sdk/.install/.backup

# Path configuration
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin

# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

# Create directories to contain code and downloaded model from GCS
RUN mkdir /root/trainer

RUN mkdir /root/models

COPY trainer/fastai_test.py /root/trainer/fastai_test.py

# Authentificate to GCP
CMD gcloud auth login

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python3",  "trainer/fastai_test.py"]
