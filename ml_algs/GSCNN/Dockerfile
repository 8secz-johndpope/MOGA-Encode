FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel

RUN apt-get -y update
RUN apt-get install -y tzdata
RUN apt-get install -y software-properties-common
RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test
RUN apt-get -y update
RUN apt-get -y upgrade

RUN apt-get install -y software-properties-common wget make \
    git curl vim vim-gnome vim htop pkg-config nano

# Install from pip
RUN pip install pyyaml \
                 scipy==1.2.0 \
                 numpy \
                 tensorflow==1.14.0 \
                 scikit-learn \
                 scikit-image \
                 matplotlib \
                 opencv-python \
                 torch==1.1.0 \
                 torchvision==0.2.0 \
                 torch-encoding==1.0.1 \
                 tensorboardX \
                 tqdm \
                 flask

COPY . /ml
WORKDIR /ml
