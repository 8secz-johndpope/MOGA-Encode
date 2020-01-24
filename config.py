# By: Oscar Andersson 2019
import json, logging
logger = logging.getLogger('gen-alg')

'''
Contains constants and "static" variables which most modules use/share.
Constants are all caps, variables are all lower case.
'''

LOG_PATH = "/output/logs/"
CLI_VERBOSITY = "INFO"  # ERROR, WARNING, INFO, DEBUG

# gen-alg parameters
POP_SIZE = 4 * 4    # must be a multiple of 4 and larger than 5!
NO_GENERATIONS = 10
MOG_ALGS = ["nsga2"]   # nsga2, nspso, moead
EPOCHS = 2
PLOT_PATH = "/output/results/"
FITNESS_DATA_PATH = "/output/results/"
ML_PERFORMANCE_BASELINE = 0.76146 #TODO: update when using big model

# optimization_problem parameters
ML_DATA_INPUT = "/data/untouched_big/"
ML_DATA_OUTPUT = "/data/cityscapes/leftImg8bit/val/"
VIDEO_ENCODERS = ["h264_nvenc", "hevc_nvenc", "libx264", "libx265", "libvpx-vp9"] # TODO: Add vaapi codecs

# ffmpeg_utils parameters
TEMP_STORAGE_PATH = "/tmp/temp.mp4" # change to tmp/temp.mp4 to use system drive instead of /tmp - tmpfs mount
IMAGE_TYPE = "png"
NAMING_SCHEME =  '%06d'  # imgtype=png & scheme='%d' --> 1.png, 2.png, 3.png...
IMG_COMP_LVL = 1


# rest_communication parameters
ML_ADDRESS = "http://localhost:5001"
REQUEST_ADDRESS = ML_ADDRESS + "/eval" 


# Global variables
timestamp = None
epoch = None
mog_alg = None

# Optimisation parameters
opt_params = None
opt_high_bounds = None
opt_low_bounds = None
opt_type = None
opt_cat_values = None
video_encoder = None
no_continous = None

def load_params_from_json(video_codec):
    '''
    Load optimisation parameters and bounds from dictionary
    '''
    global opt_params, opt_high_bounds, opt_low_bounds, opt_cat_values, video_encoder, opt_type, no_continous

    JSON_PARAM_PATH = "encoding_parameters/" + video_codec + "-parameters.json"
    video_encoder = video_codec

    opt_params = []
    opt_high_bounds = []
    opt_low_bounds = []
    opt_type = []
    opt_cat_values = {}
    no_continous = 0

    json_params = None
    with open(JSON_PARAM_PATH, 'r') as f:
        json_params = json.load(f)
    
    param_bounds = json_params["bounds"]
    opt_cat_values = json_params["categorical"]

    for param in param_bounds:
        opt_params.append(param)
        opt_low_bounds.append(param_bounds[param][0])
        opt_high_bounds.append(param_bounds[param][1])
        opt_type.append(param_bounds[param][2])
        if param_bounds[param][2] == "f": no_continous+=1
    logger.debug("Params loaded: " + str(opt_params))



# Testing degrading quality with constant parameters
ENCODER_DICT = ["libx264", "hevc_nvenc", "libvpx-vp9"]
SWEETSPOT_PARAMS = {
    "libx264": [2,2,2,1],
    "hevc_nvenc": [1,1],
    "libvpx-vp9": [2,2,2]
}