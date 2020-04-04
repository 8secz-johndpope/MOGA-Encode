# By: Oscar Andersson 2019

# Imports
import pygmo as pyg
import numpy as np
import logging, argparse, csv, re
from datetime import datetime

# Import modules
import config.config as cfg
from optimization_problem import sweetspot_problem



def evaluate_vectors_in_csv(csvpath, codec_arg, rate_control_arg):

    cfg.epoch = "0"
    cfg.mog_alg = "extended"
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

    # Evaluate decision vectors
    for pset in param_sets:
        logger.debug("Evaluating: " + str(pset))
        pop.push_back(pset)



if(__name__ == "__main__"):

    # Create timestamp used for logging and results
    cfg.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cfg.configure_logging()
    logger = logging.getLogger('gen-alg')

    parser = argparse.ArgumentParser(description='Evaluate decision vectors in CSV-file')
    parser.add_argument('-c', '--codec', default=None, help="Evaluate a specific codec")
    parser.add_argument('-rc', '--ratecontrol', default=None, help="Evaluate a specific rate control")
    parser.add_argument('-f', '--filepath', default=None, help="CSV-file path")
    args = parser.parse_args()

    evaluate_vectors_in_csv(args.filepath, args.codec, args.ratecontrol)
    