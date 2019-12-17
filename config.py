# By: Oscar Andersson 2019

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
OPTIMISATION_PARAMETERS = {  #TODO: Add more parameters
    "bitrate": [1, 50],
    "speed": [1, 3],
    "profile": [1, 3],
    "tune": [1, 4]
}
VIDEO_ENCODER = "libx264" # "h264_nvenc" "libx264"

OPT_PARAMS = None      # Is set during gen-alg initiation
OPT_HIGH_BOUNDS = None # Is set during gen-alg initiation
OPT_LOW_BOUNDS = None  # Is set during gen-alg initiation

# ffmpeg_utils parameters
TEMP_STORAGE_PATH = "/tmp/temp.mp4"
IMAGE_TYPE = "png"
NAMING_SCHEME =  '%06d'  # imgtype=png & scheme='%d' --> 1.png, 2.png, 3.png...
IMG_COMP_LVL = 1


# rest_communication parameters
ML_ADDRESS = "http://localhost:5001"
REQUEST_ADDRESS = ML_ADDRESS + "/eval" 

# These are global variables, DO NOT CHANGE
timestamp = None
epoch = None


def load_params_from_dict():
    '''
    Load optimisation parameters and bounds from dictionary
    '''
    global OPT_PARAMS, OPT_HIGH_BOUNDS, OPT_LOW_BOUNDS
    params = []
    low_bounds = []
    high_bounds = []

    for param in OPTIMISATION_PARAMETERS:
        params.append(param)
        low_bounds.append(OPTIMISATION_PARAMETERS[param][0])
        high_bounds.append(OPTIMISATION_PARAMETERS[param][-1])

    OPT_PARAMS = params
    OPT_HIGH_BOUNDS = high_bounds
    OPT_LOW_BOUNDS = low_bounds

load_params_from_dict()


# Testing degrading quality with constant parameters
ENCODER_DICT = ["libx264", "hevc_nvenc", "libvpx-vp9"]
SWEETSPOT_PARAMS = {
    "libx264": [2,2,2],
    "hevc_nvenc": [2,2,2],
    "libvpx-vp9": [2,2,2]
}