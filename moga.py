# By: Oscar Andersson 2019

# Imports
import pygmo as pyg
import numpy as np
import random, logging, os, argparse, csv
from datetime import datetime

# Import modules
import config as cfg
import plotting as pl
from optimization_problem import sweetspot_problem

# Global logger object
logger = None


def get_optimization_algorithm(randseed):
    '''
    Returns an optimisation algorithm
    '''
    opt_alg = None
    if(cfg.mog_alg == "nsga2"):
        # cr: crossover probability, m: mutation probability
        # eta_c: distribution index for crossover, eta_m: distribution index for mutation
        opt_alg = pyg.algorithm( pyg.nsga2(gen=cfg.NO_GENERATIONS, cr=0.925, m=0.05,
                                           eta_c=10, eta_m=50, seed=randseed) )
    elif(cfg.mog_alg == "moead"):
        opt_alg = pyg.algorithm ( pyg.moead(gen = cfg.NO_GENERATIONS, weight_generation = "grid",
                                            decomposition = "tchebycheff", neighbours = 5,
                                            CR = 1, F = 0.5, eta_m = 20, realb = 0.9,
                                            limit = 2, preserve_diversity = True) )
    elif(cfg.mog_alg == "nspso"):
        opt_alg = pyg.algorithm ( pyg.nspso(gen = cfg.NO_GENERATIONS, omega = 0.6, c1 = 0.01, c2 = 0.5, chi = 0.5,
                                            v_coeff = 0.5, leader_selection_range = 2,
                                            diversity_mechanism = "crowding distance",
                                            memory = False) )
    opt_alg.set_verbosity(1)
    return opt_alg


def uniform_init(prob, pop):
    '''
    Initialize chromosomes of pop with uniform initialization of all genes
    '''
    logger.debug("Populating population...")
    low_bounds, high_bounds = prob.get_bounds()

    # Create uniform decision vectors
    decision_vectors = []
    for param in range(0, len(low_bounds)):
        stepsize = (high_bounds[param]-low_bounds[param]) / cfg.POP_SIZE
        param_vals = []
        for i in range(0, cfg.POP_SIZE):
            param_vals.append(low_bounds[param] + stepsize*i)
        random.shuffle(param_vals)

        # Round param values if paramtype isnt float
        if cfg.opt_type[param] != "f":
            param_vals = np.around(param_vals, decimals=0)
        decision_vectors.append(param_vals)

    decision_vectors = np.transpose(decision_vectors)
    logger.debug("Initial decision vectors:\n" + str(decision_vectors))

    # Add decision vectors to population
    for d_vector in decision_vectors:
        logger.debug("Pushing chromosome: " + str(d_vector))
        pop.push_back(d_vector)
    return pop



def sweetspot_search(codec_arg, rate_control_arg, moga_arg):
    '''
    Evolves a population with a given optimization algorithm
    '''

    # Change optimisation config if a specific argument is set
    if codec_arg is not None: cfg.VIDEO_ENCODERS = [codec_arg]
    if rate_control_arg is not None: cfg.RATE_CONTROL[codec_arg] = [rate_control_arg]
    if moga_arg is not None: cfg.MOG_ALGS = [moga_arg]
    

    # Evaluate each codec in VIDEO_ENCODERS list
    for codec in cfg.VIDEO_ENCODERS:

        for rate_control in cfg.RATE_CONTROL[codec]:
            # Load parameters for codec
            cfg.load_params_from_json(codec, rate_control)
            logger.info("Optimising " + codec + " using " + rate_control + "...")

            for epoch in range(1, cfg.EPOCHS+1):

                for alg in cfg.MOG_ALGS:
                    cfg.epoch = epoch
                    cfg.mog_alg = alg
                    logger.info("----------- EPOCH: " + str(epoch) + " Alg: " + alg + " ------------")

                    # Get optimization problem and algorithm
                    opt_prob = pyg.problem(sweetspot_problem())

                    # Initiate population
                    rand_seed = epoch*3+1  # An arbitrary but repeatable randomisation seed
                    # random.seed(rand_seed)
                    pop = pyg.population(prob=opt_prob, seed=rand_seed)
                    pop = uniform_init(opt_prob, pop)

                    # Set up optimization algorithm
                    opt_alg = get_optimization_algorithm(rand_seed)

                    # Evolve pop using opt_alg
                    logger.debug("Starting evolution process")
                    pop = opt_alg.evolve(pop)



def get_csv_data(recon_info):
    '''
    Reconstructs a population from a CSV file
    '''
    logger.debug("Repopulating...")
    # recon_info <- [csv-file, start-index, end-index]

    with open( recon_info[0], mode='r') as csv_file:
        data = csv.reader(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        pop_data = []
        fitness_dict = {}
        for row in data:
            index = int(row[0])
            fit = [float(row[6]), float(row[7])]
            d_vector = row[4]
            d_vector = d_vector.strip('\n').strip('[').strip(']')
            x_id = d_vector
            d_vector = d_vector.split(',')
            d_vector = np.array(d_vector, dtype="float32")
            fitness_dict[x_id] = fit
            if(recon_info[1] <= index <= recon_info[2]):
                pop_data.append([d_vector, fit])

        return fitness_dict, pop_data

    logger.critical("Error loading data csv")
    exit(1)


def repopulate_pop(pop, pop_data):
    for d_vector_data in pop_data:
        x = d_vector_data[0]
        fit = d_vector_data[1]
        pop.push_back(x, fit)
        logger.debug("Pushing chromosome: " + str(x)+ "\nwith fitness: " + str(fit))
    return pop

def resume_optimisation(codec_arg, moga_arg, recon_info):

    # Load parameters for codec
    cfg.load_params_from_json(codec_arg)

    cfg.epoch = 1  # Arbitrary epoch nr
    cfg.mog_alg = moga_arg
    cfg.NO_GENERATIONS = recon_info[3]
    rand_seed = 1

    # Get optimization problem and algorithm
    fitness_dict, pop_data = get_csv_data(recon_info)
    ssp = sweetspot_problem()
    ssp.fitness_dict = fitness_dict
    opt_prob = pyg.problem(ssp)
    pop = pyg.population(prob=opt_prob)
    pop = repopulate_pop(pop, pop_data)
    logger.info("Loaded population:\n")
    logger.info(pop)
    # Set up optimization algorithm
    opt_alg = get_optimization_algorithm(rand_seed)

    # Evolve pop using opt_alg
    logger.debug("Starting evolution process")
    pop = opt_alg.evolve(pop)


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
    parser.add_argument('-a', '--moga', default=None, help="Evaluate using specific ML-algorithm")
    parser.add_argument('-c', '--codec', default=None, help="Evaluate a specific codec")
    parser.add_argument('-rc', '--ratecontrol', default=None, help="Evaluate a specific rate control")
    parser.add_argument('-r', '--resume', default=None, help="Resume optimisation using CSV-file")
    parser.add_argument('-rs', '--rstart', type=int, default=None, help="Start index of generation to resume")
    parser.add_argument('-re', '--rend', type=int, default=None, help="End index of generation to resume")
    parser.add_argument('-rg', '--rgen', type=int, default=None, help="Number of generations left to evovle")


    args = parser.parse_args()

    if(args.resume == None):
        # Start sweetspot search
        sweetspot_search(args.codec, args.ratecontrol, args.moga)
    else:
        if(args.rstart == None and
           args.rend == None and
           args.rgen == None and
           args.moga == None and
           args.codec == None):
            logger.error("Resumption of optimisation requires -rs and -re to be set with the start and end index for generation to resume")
            exit(1)
        resume_optimisation(args.codec, args.moga, [args.resume, args.rstart, args.rend, args.rgen])
