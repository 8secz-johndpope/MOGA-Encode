# By: Oscar Andersson 2019

# Imports
import numpy as np
import ffmpeg_utils # import functions from ffmpeg_utils.py
import random, logging, os
from datetime import datetime
import os, time, random, csv
import rest_communication

# Import modules
import config as cfg
import plotting as pl
import shutil
from optimization_problem import sweetspot_problem

# Global logger object
logger = None
start_bitrate, end_bitrate = 10, 1 
runs = 4



def degrade_eval():
    stepsize = (start_bitrate-end_bitrate) // runs

    original_img_size = ffmpeg_utils.get_directory_size(cfg.ML_DATA_INPUT)

    for encoder in cfg.ENCODER_DICT:
        # Load the parameters for encoder
        cfg.JSON_PARAM_PATH = encoder+"-parameters.json"
        cfg.load_params_from_json()

        params = cfg.SWEETSPOT_PARAMS[encoder]

        fitness = []

        for i in range(1, runs+1):
            # Create a decision vector to pass to ffmpeg_utilities
            decision_vector = [start_bitrate-stepsize*i]
            decision_vector += params

            # List all clips to degrade
            clips = []
            for clip in os.listdir(cfg.ML_DATA_INPUT):
                if os.path.isdir(os.path.join(cfg.ML_DATA_INPUT, clip)):
                    clips.append(clip)

            # Remove any earlier degraded clips
            for clip in clips:
                try:
                    output_clip_dir = cfg.ML_DATA_OUTPUT + clip
                    shutil.rmtree(output_clip_dir, ignore_errors=False)
                except:
                    logger.warning("Directory: " + str(cfg.ML_DATA_OUTPUT + clip) + " not found, moving on...")

            comp_size = 0

            # Apply degredation to every clip
            for clip in clips:
                input_clip_dir = cfg.ML_DATA_INPUT + clip
                output_clip_dir = cfg.ML_DATA_OUTPUT + clip
                logger.debug("Applying degredation to clip: " + input_clip_dir)
                comp_size += ffmpeg_utils.transcode(input_clip_dir, output_clip_dir, decision_vector, encoder)

            # Retrieve fitness results
            score = float(rest_communication.get_eval_from_ml_alg())    # Get ML-algorithm results 
            comp_ratio = original_img_size/comp_size               # Calc compression-ratio
            logger.info("ML-performance: " + str(score) + "\nComp-ratio: " + str(comp_ratio))

            fitness.append([-score, -comp_ratio])

        pl.plot_front_from_fitness(fitness, "Fitness of " + encoder)



def configure_logging():
    '''
    Configures the logging of information in logfiles and in CLIs.
    '''
    global logger

    # Create directory for storing results from this session
    if not os.path.isdir(cfg.PLOT_PATH + cfg.timestamp):
        os.mkdir(cfg.PLOT_PATH + cfg.timestamp)

    # create logger
    logger = logging.getLogger('gen-alg')
    logger.setLevel(logging.DEBUG)

    # Logfiles will contain full debug information
    file_log = logging.FileHandler(filename=cfg.LOG_PATH+cfg.timestamp+'.log')
    file_log.setLevel(logging.DEBUG)   
    file_log.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # Less information is printed into the console
    cli_log = logging.StreamHandler()
    cli_log.setLevel(cfg.CLI_VERBOSITY)
    cli_log.setFormatter(logging.Formatter('%(message)s'))

    # Add handlers to 'gen-alg' logger
    logger.addHandler(file_log)
    logger.addHandler(cli_log)


if(__name__ == "__main__"):
    '''
    The main function which starts the entire optimisation algorithm
    '''
    
    # Create timestamp used for logging and results
    cfg.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    configure_logging()

    # Start degredation process
    degrade_eval()
