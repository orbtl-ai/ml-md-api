# Official Tensorflow docker base image
FROM tensorflow/tensorflow:2.4.1-gpu

# Metadata
LABEL maintainer="winansw@hawaii.edu"
LABEL version="0.1"
LABEL description="This is a custom Docker image to install the NOAA Machine Learning of Marine Debris API."

# Ensure the OS doesn't attempt to prompt the user during installation of dependencies
ARG DEBIAN_FRONTEND=noninteractive

# Install OS dependencies
RUN apt-get update -y && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    gpg-agent \
    python3-opencv \
    python3-cairocffi \
    protobuf-compiler \
    python3-pil \
    python3-lxml \
    python3-tk \
    wget

RUN mkdir -p /tensorflow/models
RUN git clone https://github.com/tensorflow/models.git /tensorflow/models

# Compile protobuf configs
RUN (cd /tensorflow/models/research/ && protoc object_detection/protos/*.proto --python_out=.)
WORKDIR /tensorflow/models/research/

RUN cp object_detection/packages/tf2/setup.py ./
ENV PATH="/tensorflow/.local/bin:${PATH}"

# First copy the requirements.txt to leverage Docker cache
COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Copy over the app
COPY . /app

# Add new user to avoid running as root
RUN useradd -ms /bin/bash tensorflow
USER tensorflow
#WORKDIR /home/tensorflow

ENTRYPOINT [ "python3" ]

CMD [ "server.py" ]