# By: Oscar Andersson 2019

# Imports
import pygmo as pyg
import numpy as np
from tqdm import trange
import logging, argparse, csv, re, os

# Import modules
import config.config as cfg
from optimization_problem import sweetspot_problem


def evaluate_vectors_in_csv(csvpath, codec_arg, rate_control_arg):
    print("Evaluating vectors of: " + codec_arg + " " + rate_control_arg)

    cfg.mog_alg = "eval-from-csv"
    cfg.epoch = 0
    cfg.NO_GENERATIONS = 0
    cfg.load_params_from_json(codec_arg, rate_control_arg)
    opt_prob = pyg.problem(sweetspot_problem())
    pop = pyg.population(prob=opt_prob)

    param_sets = []
    with open(csvpath) as csv_file:
        csv_data = csv.reader(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in csv_data:
            x = row[0]
            x = x.replace('\n', '').replace('[', '').replace(']', '').replace("'", "").replace('"', "")
            x = re.sub('\s+', ',', x.strip())
            x = x.split(",")
            x = [elem for elem in x if elem != ""]
            param_sets.append(x)
    param_sets = np.asfarray(param_sets,float)

    cfg.POP_SIZE = len(param_sets)

    # Evaluate decision vectors
    for i in trange(len(param_sets)):
        pset = param_sets[i]
        pop.push_back(pset)


def eval_dir(path):
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
        cfg.configure_logging()
        evaluate_vectors_in_csv(file_path, codec_arg, rate_control_arg)




if(__name__ == "__main__"):

    logger = logging.getLogger('gen-alg')
    cfg.CLI_VERBOSITY = "ERROR"

    parser = argparse.ArgumentParser(description='Evaluate decision vectors in CSV-file')
    parser.add_argument('-c', '--codec', default=None, help="Evaluate a specific codec")
    parser.add_argument('-rc', '--ratecontrol', default=None, help="Evaluate a specific rate control")
    parser.add_argument('-f', '--filepath', default=None, help="CSV-file path")
    parser.add_argument('-ed', '--eval_dir', action="store_true", help="Evaluate directory of CSV-files")
    args = parser.parse_args()

    if args.eval_dir:
        eval_dir(args.filepath)
        exit(0)

    # Create timestamp used for logging and results
    cfg.configure_logging()
    evaluate_vectors_in_csv(args.filepath, args.codec, args.ratecontrol)
    