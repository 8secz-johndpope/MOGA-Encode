# By: Oscar Andersson 2019

# Imports
import numpy as np
import config as cfg
import plotting as pl
import random, logging, os, argparse, csv, re, json, shutil
from datetime import datetime

import ffmpeg_utils # import functions from ffmpeg_utils.py
import rest_communication

# Global logger object
logger = None


def degrade_eval(codec_arg, rate_control_arg, csvpath):

    # Set codec and rate control of which the coding parameters are for
    cfg.VIDEO_ENCODERS = [codec_arg]
    cfg.RATE_CONTROLS[codec_arg] = [rate_control_arg]


    # List all scenarios to degrade
    scenarios = []
    for scenario in os.listdir(cfg.ML_DATA_INPUT):
        if os.path.isdir(os.path.join(cfg.ML_DATA_INPUT, scenario)):
            scenarios.append(scenario)


    ## Load param info from json specific for codec and rate control
    cfg.load_params_from_json(codec_arg, rate_control_arg)


    # CSV loading
    param_sets = []
    with open(csvpath) as csv_file:
        csv_data = csv.reader(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in csv_data:
            x = str(row[0])
            x = x.replace('\n', '').replace('[', '').replace(']', '')
            x = re.sub('\s+', ',', x.strip())
            param_sets.append(x.split(","))

    param_sets = np.asfarray(param_sets,float)

    scenario_results = {}
    
    for scenario in scenarios:
        logger.info("Evaluating scenario: " + str(scenario))

        input_scenario_dir = cfg.ML_DATA_INPUT +"/"+ scenario
        output_scenario_dir = cfg.ML_DATA_OUTPUT +"/situation"
        original_scenario_size = ffmpeg_utils.get_directory_size(input_scenario_dir)

        results = {}
        for param_set in param_sets:

            logger.info("Evaluating parameter set: " + str(param_set))

            # Remove any earlier degraded scenarios
            try:
                shutil.rmtree(output_scenario_dir, ignore_errors=False)
            except:
                logger.warning("Directory: " + str(cfg.ML_DATA_OUTPUT +"/situation") + " not found, moving on...")

            # Compress dataset and retrieve its size
            comp_size = ffmpeg_utils.transcode(input_scenario_dir, output_scenario_dir, param_set)
            
            # TODO: add SSMI/PSNR eval!
            
            # Retrieve fitness results
            _, full_response = rest_communication.get_eval_from_ml_alg(eval_list=scenario)    # Get ML-algorithm results 
            comp_ratio = original_scenario_size/comp_size               # Calc compression-ratio
            results[str(param_set)] = [full_response, comp_ratio]

        scenario_results[scenario] = results
    
    with open(cfg.FITNESS_DATA_PATH + cfg.timestamp + "/" + "degrade-results.json", mode='w') as data_file:
        json.dump(scenario_results, data_file)

    # SAVE CSV RESULTS!!
    # TODO: Fix formatting for easy excell import 
    for key in scenario_results.keys():
        with open(cfg.FITNESS_DATA_PATH + cfg.timestamp + '/'+ key + '-results.csv', mode='w') as data_file:
            scenario = scenario_results[key] 
            data_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            data = np.concatenate((key, scenario_results[key]), axis=None)
            data_writer.writerow(data)



def configure_logging():
    '''
    Configures the logging of information in logfiles and in CLIs.
    '''
    global logger

    # Check/prepare directories for storing data from this session
    if not os.path.isdir(cfg.PLOT_PATH): os.mkdir(cfg.PLOT_PATH) 
    if not os.path.isdir(cfg.LOG_PATH): os.mkdir(cfg.LOG_PATH)
    if not os.path.isdir(cfg.PLOT_PATH + cfg.timestamp): os.mkdir(cfg.PLOT_PATH + cfg.timestamp)

    # create logger
    logger = logging.getLogger('gen-alg')
    logger.setLevel(logging.DEBUG)

    # Logfiles will contain full debug information
    file_log = logging.FileHandler(filename=cfg.LOG_PATH+cfg.timestamp+'.log')
    file_log.setLevel(logging.DEBUG)   
    file_log.setFormatter(logging.Formatter('Time: %(asctime)s  Level: %(levelname)s\n%(message)s\n'))

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

    parser = argparse.ArgumentParser(description='Multi objective genetic algorithm')
    parser.add_argument('-c', '--codec', default=None, help="Codec of evaluation")
    parser.add_argument('-rc', '--ratecontrol', default=None, help="Rate control of evaluation")
    parser.add_argument('-f', '--csvfile', default=None, help="CSV-file with coding parameters to evaluate")

    args = parser.parse_args()

    # Check that input has been given
    if( args.csvfile == None and
        args.codec == None and
        args.ratecontrol == None ):
        logger.error("Arguments missing!")
        exit(1)

    # Start degredation process
    degrade_eval(args.codec, args.ratecontrol, args.csvfile)
