# MOGA-Encode
Multi Objective Genetic Algorithm for finding optimal encoding setups

## Requirements
Docker CE, Docker-compose and nvidia-container-runtime

## Setup
### Step 1 - clone and build

- clone this repository (MOGA-Encode)
- extract the ML-data into the parent directory (../*"ML-rep"*-mldata)
- Clone the ML-repository into the parent directory of MOGA-Encode
- cd into MOGA-Encode directory
- run: docker-compose build


### Step 2 - running services:

- cd into MOGA-Encode directory
- run: docker-compose run -d *"ml-alg-service"*
- wait 10 seconds
- run: docker-compose run -d moga-encode

**NOTE!**

To properly run VAAPI encoding, verify that i965 is used!! (export LIBVA_DRIVER_NAME=i965 otherwise) 

#### Debugging

**Console 1**

- cd into *ml-alg-service* directory
- run: docker-compose run *"ml-alg-service"*

**Console 2**

- cd into MOGA-Encode directory
- run: docker-compose run moga-encode
- inside moga-encode, run: python3 moga.py


### Stopping services:

Run: docker-compose down
