FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel

# Install from pip
RUN apt-get -y update
RUN apt-get install -y tzdata
RUN apt -y upgrade

RUN apt install -y software-properties-common wget make git curl htop libsm6 libxext6 libxrender-dev p7zip-full ninja-build nano

RUN conda install -c anaconda pip

#RUN conda install -c conda-forge opencv
RUN pip install numpy \
                EasyDict==1.7 \
                opencv-python==3.4.1.15 \
                shapely==1.6.4 \
                torchvision \
                Cython \
                scipy \
                pandas \
                pyyaml \
                json_tricks \
                scikit-image \
                yacs \
                tensorboardX \
                tqdm \
                flask

COPY . /ml
WORKDIR /ml
