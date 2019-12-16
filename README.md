# MOGA-Encode
Multi Objective Genetic Algorithm for finding optimal encoding setups

## Requirements
Docker CE, Docker-compose and nvidia-container-runtime

## Setup
### Step 1

- clone this repository
- extract the ML-data into the parent directory

### Step 2 - get images

### Alt 1 - load images from archives:

- download archives of hrnet and gen-alg
- run: docker load < mlalg.tar.gz
- run: docker load < moga.tar.gz 

### Alt 2 - build from source

- Clone the ML-repository into the parent directory
- cd into MOGA-Encode directory
- run: docker-compose build


### Step 2 - run services:

#### Console 1

- cd into MOGA-Encode directory
- run: docker-compose run hrnet

#### Console 2

- cd into MOGA-Encode directory
- run: docker-compose run moga-encode
- inside moga-encode, run: python3 moga.py



