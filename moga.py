# By: Oscar Andersson 2019

# Imports
import pygmo as pyg
import numpy as np
import random, logging, os, argparse, csv, pickle
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
        opt_alg = pyg.algorithm( pyg.nsga2(gen=1, cr=0.925, m=0.05,
                                           eta_c=10, eta_m=50, seed=randseed) )
    elif(cfg.mog_alg == "moead"):
        opt_alg = pyg.algorithm ( pyg.moead(gen = 1, weight_generation = "grid",
                                            decomposition = "tchebycheff", neighbours = 5,
                                            CR = 1, F = 0.5, eta_m = 20, realb = 0.9,
                                            limit = 2, preserve_diversity = True) )
    elif(cfg.mog_alg == "nspso"):
        opt_alg = pyg.algorithm ( pyg.nspso(gen = 1, omega = 0.6, c1 = 0.01, c2 = 0.5, chi = 0.5,
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
    if rate_control_arg is not None: cfg.RATE_CONTROLS[codec_arg] = [rate_control_arg]
    if moga_arg is not None: cfg.MOG_ALGS = [moga_arg]
    

    # Evaluate each codec in VIDEO_ENCODERS list
    for codec in cfg.VIDEO_ENCODERS:

        for rate_control in cfg.RATE_CONTROLS[codec]:
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
                    for gen in range(1, cfg.NO_GENERATIONS):
                        logger.info("Generation: " + str(gen+1))
                        pop = opt_alg.evolve(pop)
                        pickle.dump( pop, open( cfg.POPULATION_PICKLE_PATH, "wb" ) )                    



def resume_optimisation(codec, rate_control, moga, base_gen, epoch):

    cfg.VIDEO_ENCODERS = [codec]
    cfg.RATE_CONTROLS[codec] = [rate_control]
    cfg.MOG_ALGS = [moga]
    
    # Load parameters for codec
    cfg.load_params_from_json(codec, rate_control)
    logger.info("Optimising " + codec + " using " + rate_control + "...")

    cfg.epoch = epoch
    cfg.mog_alg = moga
    
    # Initiate population
    rand_seed = epoch*3+1  # An arbitrary but repeatable randomisation seed
    # random.seed(rand_seed)

    logger.debug("Loading population from pickle")
    pop = pickle.load( open( cfg.POPULATION_PICKLE_PATH, "rb" ) )
    logger.info(pop)

    # Set up optimization algorithm
    opt_alg = get_optimization_algorithm(rand_seed)

    # Evolve pop using opt_alg
    logger.debug("Starting evolution process")
    for gen in range(base_gen, cfg.NO_GENERATIONS):
        logger.info("Generation: " + str(gen+1))
        pop = opt_alg.evolve(pop)
        pickle.dump( pop, open( cfg.POPULATION_PICKLE_PATH, "wb" ) )                    



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
    parser.add_argument('-r', '--resume', action="store_true", help="Resume optimisation using CSV-file")
    parser.add_argument('-rg', '--rgen', type=int, default=None, help="Number of generations already evolved")
    parser.add_argument('-re', '--repoch', type=int, default=1, help="Epoch number of resumption")


    args = parser.parse_args()

    if(not args.resume):
        # Start sweetspot search
        sweetspot_search(args.codec, args.ratecontrol, args.moga)
    else:
        if(args.rgen == None and
           args.moga == None and
           args.ratecontrol == None and
           args.codec == None):
            logger.error("Resumption of optimisation requires moga, codec, ratecontrol and rgen to be set")
            exit(1)
        resume_optimisation(args.codec, args.ratecontrol, args.moga, args.rgen, args.repoch)
