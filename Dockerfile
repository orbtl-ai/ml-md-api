# Official Tensorflow docker base image
FROM tensorflow/tensorflow:2.4.1-gpu

# Metadata
LABEL maintainer="ross@orbtl.ai"
LABEL version="0.2"
LABEL description="This is a custom Docker image to install the Machine Learning of Marine Debris API (ML/MD API)."

ADD https://github.com/uclouvain/openjpeg/archive/v2.4.0.tar.gz /usr/local/src/openjpeg-2.4.0.tar.gz

# Ensure the OS doesn't attempt to prompt the user during installation of dependencies
ARG DEBIAN_FRONTEND=noninteractive

# Install OS dependencies
RUN apt-get update -y && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    libjpeg-dev \
    gpg-agent \
    python3-opencv \
    python3-cairocffi \
    protobuf-compiler \
    #python3-pil \
    python3-lxml \
    python3-tk \
    wget \
    cmake

RUN mkdir -p /tensorflow/models
RUN git clone https://github.com/tensorflow/models.git /tensorflow/models

# Compile protobuf configs
RUN (cd /tensorflow/models/research/ && protoc object_detection/protos/*.proto --python_out=.)
WORKDIR /tensorflow/models/research/

RUN cp object_detection/packages/tf2/setup.py ./
ENV PATH="/tensorflow/.local/bin:${PATH}"

WORKDIR /usr/local/

RUN cd src && tar -xvf openjpeg-2.4.0.tar.gz && cd openjpeg-2.4.0/ \
    && mkdir build && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local/ \
    && make && make install && make clean \
    && cd /usr/local/ && rm -Rf src/openjpeg*

WORKDIR /app

# First copy the requirements.txt to leverage Docker cache
COPY ./requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Copy over the app
##COPY . /app

#ENTRYPOINT [ "python3" ]

#CMD [ "server.py" ]