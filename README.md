# MOGA-Encode
Multi Objective Genetic Algorithm for finding optimal encoding setups

## Requirements
Docker CE, Docker-compose and nvidia-container-runtime

## Setup
### Step 1 - clone and build

- clone this repository (MOGA-Encode)
- For GSCNN and HRNet: Clone the ML-repository into the ml_algs directory of MOGA-Encode, do not replace existing files under this directory
- cd into MOGA-Encode directory
- run: docker-compose build (If HRNet or GSCNN is not used, remove these from docker-compose.yml)


### Step 2 - running services (optimisation):

- configure config/config.py
- cd into MOGA-Encode directory
- run: docker-compose run -d *"ml-alg-service"*
- wait 15~ seconds
- run optimisation: docker-compose run -d moga-encode

**NOTE!**

If problems are encountered for VAAPI encoders, investigate whether i965 or iHD is used!! (setting LIBVA_DRIVER_NAME=i965 or iHD might be required) 

### Step 3 - running services (evaluation):

- cd into MOGA-Encode directory
- check options for evaluation: docker-compose run moga-encode python3 -m tools.degrade-eval -h
- configure config/config.py
- run: docker-compose run -d *"ml-alg-service"*
- wait 15~ seconds
- run evaluation: docker-compose run -d moga-encode python3 -m tools.degrade-eval [options]


#### Debugging

**Console 1**

- cd into *ml-alg-service* directory
- run: docker-compose run *"ml-alg-service"*

**Console 2**

- cd into MOGA-Encode directory
- run: docker-compose run moga-encode bash
- inside moga-encode, run: "python3 moga.py [options]" or "python3 -m tools.degrade-eval [options]"


### Stopping services:

Run: docker-compose down
