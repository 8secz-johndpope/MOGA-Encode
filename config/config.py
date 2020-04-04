# By: Oscar Andersson 2019
import json, logging, os
from datetime import datetime

logger = None

'''
Contains constants and "static" variables which most modules use/share.
Constants are all caps, variables are all lower case.
'''

OUTPUT_BASE = "output"
LOG_PATH = OUTPUT_BASE + "/logs/"
POPULATION_PICKLE_PATH = OUTPUT_BASE + "/population.p"
CLI_VERBOSITY = "INFO"  # ERROR, WARNING, INFO, DEBUG

# gen-alg parameters
POP_SIZE = 4 * 4    # must be a multiple of 4 and larger than 5!
NO_GENERATIONS = 1
MOG_ALGS = ["nsga2"]   # nsga2, nspso, moead
EPOCHS = 1
PLOT_PATH = OUTPUT_BASE + "/results/"
FITNESS_DATA_PATH = OUTPUT_BASE + "/results/"

ML_PERFORMANCE_BASELINE = 0.8162191842797382
#ML_PERFORMANCE_BASELINE = 0.806058279492062
ML_PERFORMANCE_MEASURE = "mean_IoU"
#ML_PERFORMANCE_MEASURE = "mean_iu"

ML_DATA_BASE = "data"
# optimization_problem parameters
#ML_DATA_INPUT = ML_DATA_BASE + "/Cityscapes-dataset/scenario_suites/"
#ML_DATA_INPUT = ML_DATA_BASE + "/Cityscapes-dataset/scenario_suites_gn0,0005_gb0,5"
ML_DATA_INPUT = ML_DATA_BASE + "/Cityscapes-dataset/untouched_tiny/"
ML_DATA_OUTPUT = ML_DATA_BASE + "/HRNet-mldata/cityscapes/leftImg8bit/val/"
#ML_DATA_OUTPUT = ML_DATA_BASE + "/GSCNN-mldata/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/val/"
VIDEO_ENCODERS = ["libx264"]
RATE_CONTROLS = { "h264_nvenc": ["CBR"],
                 "hevc_nvenc": ["CBR"],
                 "libx264":    ["ABR"],
                 "libx264rgb": ["ABR"],
                 "libx265":    ["ABR"],
                 "h264_vaapi": ["CBR"],
                 "hevc_vaapi": ["CQP"],
                 "vp9_vaapi": ["CBR"],
                 "libvpx-vp9": ["CBR"],
                 "libaom-av1": ["CBR"],
                 "libsvt_av1": ["CBR"] }

# ffmpeg_utils parameters
TEMP_STORAGE_PATH = "/tmp/temp.mp4" # change to tmp/temp.mp4 to use system drive instead of /tmp - tmpfs mount
IMAGE_TYPE = "png"
NAMING_SCHEME =  '%06d'  # imgtype=png & scheme='%d' --> 1.png, 2.png, 3.png...
IMG_COMP_LVL = 1

JSON_PARAM_PATH_BASE = "config/encoding_parameters"


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
opt_constants = None
video_encoder = None
rate_control = None
no_continous = None

def load_params_from_json(encoder, r_control):
    '''
    Load optimisation parameters and bounds from dictionary
    '''
    global opt_params, opt_high_bounds, opt_low_bounds, opt_cat_values, video_encoder, rate_control, opt_type, no_continous, opt_constants

    JSON_PARAM_PATH = JSON_PARAM_PATH_BASE + "/" + encoder + "-parameters.json"
    video_encoder = encoder
    rate_control = r_control

    opt_params = []
    opt_high_bounds = []
    opt_low_bounds = []
    opt_type = []
    opt_cat_values = {}
    no_continous = 0

    json_params = None
    with open(JSON_PARAM_PATH, 'r') as f:
        json_params = json.load(f)

    try:
        json_params = json_params[rate_control]
        param_bounds = json_params["bounds"]
        opt_cat_values = json_params["categorical"]
        opt_constants = json_params["constants"]
    except Exception as e:
        logger.critical("Faulty JSON file or configuration, KeyError:" + str(e))
        exit(1)

    for param in param_bounds:
        opt_params.append(param)
        opt_low_bounds.append(param_bounds[param][0])
        opt_high_bounds.append(param_bounds[param][1])
        opt_type.append(param_bounds[param][2])
        if param_bounds[param][2] == "f": no_continous+=1
    logger.debug("Params loaded: " + str(opt_params))


def configure_logging():
    '''
    Configures the logging of information in logfiles and in CLIs.
    '''
    global logger, timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Check/prepare directories for storing data from this session
    if not os.path.isdir(PLOT_PATH): os.mkdir(PLOT_PATH) 
    if not os.path.isdir(LOG_PATH): os.mkdir(LOG_PATH)
    if not os.path.isdir(PLOT_PATH + timestamp): os.mkdir(PLOT_PATH + timestamp)

    # create logger
    logger = logging.getLogger('gen-alg')
    logger.setLevel(logging.DEBUG)

    # Logfiles will contain full debug information
    file_log = logging.FileHandler(filename=LOG_PATH+timestamp+'.log')
    file_log.setLevel(logging.DEBUG)   
    file_log.setFormatter(logging.Formatter('Time: %(asctime)s  Level: %(levelname)s\n%(message)s\n'))

    # Less information is printed into the console
    cli_log = logging.StreamHandler()
    cli_log.setLevel(CLI_VERBOSITY)
    cli_log.setFormatter(logging.Formatter('%(message)s'))

    # Add handlers to 'gen-alg' logger
    logger.addHandler(file_log)
    logger.addHandler(cli_log)