# By: Oscar Andersson 2019
import json, logging
logger = logging.getLogger('gen-alg')

'''
Contains constants and "static" variables which most modules use/share.
Constants are all caps, variables are all lower case.
'''

LOG_PATH = "logs/"
CLI_VERBOSITY = "INFO"  # ERROR, WARNING, INFO, DEBUG

# gen-alg parameters
POP_SIZE = 4 * 3    # must be a multiple of 4 and larger than 5!
NO_GENERATIONS = 10  
MOG_ALG = "nsga2"   # nsga2, nspso, moead
EPOCHS = 3
PLOT_PATH = "results/"
FITNESS_DATA_PATH = "results/"
ML_PERFORMANCE_BASELINE = 0.76146 #TODO: update when using big model

# optimization_problem parameters
ML_DATA_INPUT = "/data/untouched_small/"  #TODO: use larger dataset
ML_DATA_OUTPUT = "/data/cityscapes/leftImg8bit/val/"
JSON_PARAM_PATH = "libx264-parameters.json"
VIDEO_ENCODER = "libx264" # "h264_nvenc" "libx264"


# ffmpeg_utils parameters
TEMP_STORAGE_PATH = "/tmp/temp.mp4" # change to tmp/temp.mp4 to use system drive instead of /tmp - tmpfs mount
IMAGE_TYPE = "png"
NAMING_SCHEME =  '%06d'  # imgtype=png & scheme='%d' --> 1.png, 2.png, 3.png...
IMG_COMP_LVL = 1


# rest_communication parameters
ML_ADDRESS = "http://localhost:5001"
REQUEST_ADDRESS = ML_ADDRESS + "/eval" 


# These are global variables, DO NOT CHANGE
timestamp = None
epoch = None

# Optimisation parameters
OPT_PARAMS = []
OPT_HIGH_BOUNDS = []
OPT_LOW_BOUNDS = []
OPT_VALUES = {}

def load_params_from_json():
    '''
    Load optimisation parameters and bounds from dictionary
    '''
    global OPT_PARAMS, OPT_HIGH_BOUNDS, OPT_LOW_BOUNDS, OPT_VALUES
    
    json_params = None
    with open(JSON_PARAM_PATH, 'r') as f:
        json_params = json.load(f)
    
    param_bounds = json_params["bounds"]
    OPT_VALUES = json_params["values"]

    for param in param_bounds:
        OPT_PARAMS.append(param)
        OPT_LOW_BOUNDS.append(param_bounds[param][0])
        OPT_HIGH_BOUNDS.append(param_bounds[param][1])
    logger.debug("Params loaded: " + str(OPT_PARAMS))

load_params_from_json()


# Testing degrading quality with constant parameters
ENCODER_DICT = ["libx264", "hevc_nvenc", "libvpx-vp9"]
SWEETSPOT_PARAMS = {
    "libx264": [2,2,2],
    "hevc_nvenc": [2,2,2],
    "libvpx-vp9": [2,2,2]
}