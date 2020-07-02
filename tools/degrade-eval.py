# By: Oscar Andersson 2019

# Imports
import numpy as np
import pygmo as pyg
from skimage import io, metrics
from multiprocessing import Pool
import config.config as cfg
import utils.plotting as pl
from tqdm import trange
import random, logging, os, argparse, csv, re, json, shutil, time
from datetime import datetime

import utils.ffmpeg_utils as ffu    # import functions from ffmpeg_utils.py
import utils.rest_communication as restcom # import functions from rest_communication.py

# Global logger object
logger = None
ORIG_TEST = False


def degrade_eval(codec_arg, rate_control_arg, csvpath, degradation):

    ML_DATA_INPUT = cfg.ML_DATA_EVAL_INPUT[degradation]

    # Set codec and rate control of which the coding parameters are for
    cfg.VIDEO_ENCODERS = [codec_arg]
    cfg.RATE_CONTROLS[codec_arg] = [rate_control_arg]

    # List all scenarios to degrade
    scenarios = []
    for scenario in os.listdir(ML_DATA_INPUT):
        if os.path.isdir(os.path.join(ML_DATA_INPUT, scenario)):
            scenarios.append(scenario)

    param_sets = ["orig"]
    if not ORIG_TEST:
        ## Load param info from json specific for codec and rate control
        cfg.load_params_from_json(codec_arg, rate_control_arg)

        # Loading coding parameters from CSV
        param_sets = load_param_set(csvpath)

    # Iterate through each scenario and evaluate each set of coding parameters
    logger.info("Scenarios: " + str(scenarios))
    for i in trange(len(scenarios)):

        scenario = scenarios[i]
        input_scenario_dir = ML_DATA_INPUT +"/"+ scenario
        output_scenario_dir = cfg.ML_DATA_OUTPUT +"/scenario"
        original_scenario_size = ffu.get_directory_size(input_scenario_dir)
        results = {}

        # Iterate through each set of coding parameters
        for param_set in param_sets:
            # Remove any earlier degraded scenarios
            shutil.rmtree(output_scenario_dir, ignore_errors=True)

            # Compress dataset and retrieve its size
            comp_size = 1
            mean_comparison_results = ["", "", "", ""]
            if ORIG_TEST:
                try:
                    shutil.copytree(input_scenario_dir, output_scenario_dir)
                except Exception as e:
                    logger.error(e)
                    exit(1)
            else:
                comp_size = ffu.transcode(input_scenario_dir, output_scenario_dir, param_set)
                # Comparison between original and compressed frames using mean SSMI, PSNR and other metrics
                mean_comparison_results = get_structural_comparison(input_scenario_dir, output_scenario_dir)

            # Retrieve fitness results
            _, full_response = restcom.get_eval_from_ml_alg(eval_list=scenario)    # Get ML-algorithm results 
            comp_ratio = original_scenario_size/comp_size               # Calc compression-ratio
            results[decision_vector_to_string(param_set)] = [*full_response.values(), comp_ratio, *mean_comparison_results]

        # Save scenario results to CSV-file
        with open(cfg.RESULTS_PATH+cfg.timestamp+'/'+codec_arg+'_'+rate_control_arg+'_results.csv', mode='a') as data_file:
            data_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for param_set in results.keys():
                data = results[param_set]
                data = np.concatenate((param_set, data), axis=None)
                data = np.concatenate((scenario, data), axis=None)
                data_writer.writerow(data)


def degrade_eval_dirs(path, degradation):
    csv_files = []
    for f in os.listdir(path):
        if os.path.isfile(os.path.join(path, f)):
            if f.endswith('.csv'):
                csv_files.append(f)
    
    for f in csv_files: 
        file_path = os.path.join(path, f)
        args = f.split(":")
        codec_arg = args[0]
        rate_control_arg = args[1]
        degrade_eval(codec_arg, rate_control_arg, file_path, degradation)


def get_structural_comparison(orig_path, comp_path):

    # Get the filenames of all images in the original dataset
    filenames = ffu.get_names(orig_path)
    
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


def string_to_decision_vector(x):
    x = x.replace('\n', '').replace('[', '').replace(']', '').replace("'", "").replace('"', "")
    x = re.sub('\s+', ',', x.strip())
    x = x.split(",")
    x = [elem for elem in x if elem != ""]  # Remove empty spaces
    return x


def load_param_set(csvpath):
    param_sets = []
    with open(csvpath) as csv_file:
        csv_data = csv.reader(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in csv_data:
            param_sets.append(string_to_decision_vector(str(row[0])))
    return np.asfarray(param_sets,float)



if(__name__ == "__main__"):
    '''
    The main function which starts the entire optimisation algorithm
    '''
    # Create timestamp used for logging and results
    logger = logging.getLogger('gen-alg')
    cfg.CLI_VERBOSITY = "ERROR"
    cfg.configure_logging()

    parser = argparse.ArgumentParser(description='Multi objective genetic algorithm')
    parser.add_argument('-f', '--csvfile', default=None, help="CSV-file with coding parameters to evaluate")
    parser.add_argument('-d', '--degradation', default="nondeg", help="The degradation of data: nondeg, rain, noise, moving")
    parser.add_argument('-ed', '--evaldir', action="store_true", help="evaluate csvs in directory")
    parser.add_argument('-c', '--codec', default=None, help="Codec of evaluation (used if evaldir in not set)")
    parser.add_argument('-rc', '--ratecontrol', default=None, help="Rate control of evaluation  (used if evaldir in not set)")
    parser.add_argument('--orig_test', action="store_true", help="Collect data for uncompressed dataset")
    

    args = parser.parse_args()

    if args.orig_test:
        degrade_eval("orig", "dataset", "", "nondeg")
        exit(0)


    # Check that input has been given
    if( args.evaldir):
        degrade_eval_dirs(args.csvfile, args.degradation)
        exit(0)

    # Check that input has been given
    if( args.csvfile == None and
        args.codec == None and
        args.ratecontrol == None ):
        logger.error("Arguments missing!")
        exit(1)

    # Start degredation process
    degrade_eval(args.codec, args.ratecontrol, args.csvfile, args.degradation)

