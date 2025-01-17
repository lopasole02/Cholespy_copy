FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

# Install dependencies for building Python
RUN apt-get update && apt-get install -y \
    make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncurses5-dev libncursesw5-dev xz-utils tk-dev \
    libffi-dev liblzma-dev python-openssl git 

# Install pyenv to install specific Python version
RUN curl https://pyenv.run | bash

# Set environment variables and initialize pyenv
ENV HOME /root
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH

# Initialize pyenv and install Python 3.11
RUN pyenv install 3.11.0 && pyenv global 3.11.0

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python get-pip.py

# Install Python libraries
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install specific packages for building SUPN Cholespy
RUN apt-get update && apt-get install -y \
    cmake libsuitesparse-dev g++-8

RUN pip install ninja scikit-build scikit-sparse nanobind eigen

# Set the working directory (within the container)
WORKDIR /home/supn_cholespy

# Copy your package source code to the image
COPY . /home/supn_cholespy

RUN git submodule update --init --recursive

WORKDIR /home/supn_cholespy/ext/nanobind

RUN git checkout method_bindings

WORKDIR /home/supn_cholespy

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8

RUN pip install develop .

# Clean up to reduce image size
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
