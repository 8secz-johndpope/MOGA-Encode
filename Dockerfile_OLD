FROM jrottenberg/ffmpeg:4.2-nvidia

RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get -y install python3 python3-pip wget make git nano

# pip packages
RUN pip3 install numpy \
                 pygmo \
                 matplotlib \
                 ffmpeg-python \
                 requests

COPY . /MOGA-Encode
WORKDIR /MOGA-Encode
