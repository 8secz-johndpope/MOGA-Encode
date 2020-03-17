# By: Oscar Andersson 2019

# Imports
import numpy as np
from skimage import io, metrics
from multiprocessing import Pool
import config as cfg
import plotting as pl
import random, logging, os, argparse, csv, re, json, shutil, time
from datetime import datetime

import ffmpeg_utils # import functions from ffmpeg_utils.py
import rest_communication # import functions from rest_communication.py

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

    # Loading coding parameters from CSV
    param_sets = load_param_set(csvpath)

    # Iterate through each scenario and evaluate each set of coding parameters
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
            shutil.rmtree(output_scenario_dir, ignore_errors=True)

            # Compress dataset and retrieve its size
            comp_size = ffmpeg_utils.transcode(input_scenario_dir, output_scenario_dir, param_set)
            
            # Comparison between original and compressed frames using mean SSMI, PSNR and other metrics
            mean_comparison_results = get_structural_comparison(input_scenario_dir, output_scenario_dir)

            # Retrieve fitness results
            _, full_response = rest_communication.get_eval_from_ml_alg(eval_list=scenario)    # Get ML-algorithm results 
            comp_ratio = original_scenario_size/comp_size               # Calc compression-ratio
            results[decision_vector_to_string(param_set)] = [*full_response.values(), comp_ratio, *mean_comparison_results]

        scenario_results[scenario] = results

    for key in scenario_results.keys():
        with open(cfg.FITNESS_DATA_PATH + cfg.timestamp + '/'+ key + '-results.csv', mode='w') as data_file:
            scenario = scenario_results[key] 
            data_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for param_set in scenario.keys():
                data = scenario[param_set]
                data = np.concatenate((param_set, data), axis=None)
                data = np.concatenate((key, data), axis=None)
                data_writer.writerow(data)



def get_structural_comparison(orig_path, comp_path):

    # Get the filenames of all images in the original dataset
    filenames = ffmpeg_utils.get_names(orig_path)
    
    # Cityscapes: Only evaluate the frames which are evaluated by the ML-algorithms
    reduced_set = []
    for i in range(0, len(filenames)//30):
        reduced_set.append(filenames[i*30 + 19])
    filenames = reduced_set
    
    # Use pool of workers to evaluate images in parallel
    pool = Pool(processes=16)
    calc_map = [ (os.path.join(orig_path, filename),
                  os.path.join(comp_path, filename))
                  for filename in filenames ]
    res_arr = pool.map(parallel_comparison, calc_map)
    pool.terminate()

    # Calculate the mean values from evals
    mean_results = np.mean(res_arr, axis=0)
    return mean_results


def parallel_comparison(args):
    orig_path, comp_path = args
    orig_img = io.imread(orig_path)
    comp_img = io.imread(comp_path)
    comparisons = []
    comparisons.append(metrics.structural_similarity(orig_img, comp_img, multichannel=True))
    comparisons.append(metrics.peak_signal_noise_ratio(orig_img, comp_img))
    comparisons.append(metrics.mean_squared_error(orig_img, comp_img))
    comparisons.append(metrics.normalized_root_mse(orig_img, comp_img))
    return comparisons


def decision_vector_to_string(d_vector):
    string = ""
    for vector in d_vector:
        string += str(vector)
        string += ","
    string = string[:-1]
    return string


def load_param_set(csvpath):
    param_sets = []
    with open(csvpath) as csv_file:
        csv_data = csv.reader(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in csv_data:
            x = str(row[0])
            x = x.replace('\n', '').replace('[', '').replace(']', '')
            x = re.sub('\s+', ',', x.strip())
            param_sets.append(x.split(","))

    return np.asfarray(param_sets,float)


def convert_param_set(csvpath):
    eval_sets = []
    with open(csvpath) as csv_file:
        csv_data = csv.reader(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in csv_data:
            x = str(row[0])
            x = x.replace('\n', '').replace('[', '').replace(']', '')
            x = re.sub('\s+', ',', x.strip())
            row[0] = decision_vector_to_string( np.asfarray(x.split(","), float) )
            eval_sets.append(row)

    with open(csvpath + "_new.csv", mode='w') as data_file:
        data_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for es in eval_sets:
            data_writer.writerow(es)


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
    parser.add_argument('--convert', action="store_true", help="Convert older CSV-files to more readable format")

    args = parser.parse_args()

    if(args.convert == True):
        convert_param_set(args.csvfile)
        exit(0)

    # Check that input has been given
    if( args.csvfile == None and
        args.codec == None and
        args.ratecontrol == None ):
        logger.error("Arguments missing!")
        exit(1)

    # Start degredation process
    degrade_eval(args.codec, args.ratecontrol, args.csvfile)
