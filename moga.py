# By: Oscar Andersson 2019

# Imports
import pygmo as pyg
import numpy as np
import random, logging, os, argparse
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
        opt_alg = pyg.algorithm( pyg.nsga2(gen=cfg.NO_GENERATIONS, cr=0.925, m=0.1,
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


def uniform_bitrate_init(prob, pop):
    '''
    Initialize chromosomes of pop with uniform initialization of bitrate gene
    and random initialization for the other genes
    '''
    logger.debug("Populating population...")
    low_bounds, high_bounds = prob.get_bounds()
    stepsize = (high_bounds[0]-low_bounds[0]) / cfg.POP_SIZE
    for i in range(0, cfg.POP_SIZE):
        bitrate = round(low_bounds[0] + stepsize*i, 3)
        x = [bitrate]
        for j in range(1, len(low_bounds)):
            if cfg.opt_type[j] == "f":
                x.append(random.uniform(low_bounds[j], high_bounds[j]))
            else:
                x.append(random.randint(low_bounds[j], high_bounds[j]))
        logger.debug("Pushing chromosome: " + str(x))
        pop.push_back(x)
    return pop



def sweetspot_search(codec_arg, moga_arg):
    '''
    Evolves a population with a given optimization algorithm
    '''

    # Change optimisation config if a specific argument is set
    if codec_arg is not None: cfg.VIDEO_ENCODERS = [codec_arg]
    if moga_arg is not None: cfg.MOG_ALGS = [moga_arg]


    # Evaluate each codec in VIDEO_ENCODERS list
    for codec in cfg.VIDEO_ENCODERS:
        logger.info("Optimising the " + codec + " video-codec...")

        # Load parameters for codec
        cfg.load_params_from_json(codec)

        for epoch in range(1, cfg.EPOCHS+1):

            for alg in cfg.MOG_ALGS:
                cfg.epoch = epoch
                cfg.mog_alg = alg
                logger.info("----------- EPOCH: " + str(epoch) + " Alg: " + alg + " ------------")

                # Get optimization problem and algorithm
                opt_prob = pyg.problem(sweetspot_problem())

                # Initiate population
                rand_seed = epoch*3+1  # An arbitrary but repeatable randomisation seed
                pop = pyg.population(prob=opt_prob, seed=rand_seed)
                pop = uniform_bitrate_init(opt_prob, pop)

                # Set up optimization algorithm
                opt_alg = get_optimization_algorithm(rand_seed)

                # Evolve pop using opt_alg
                logger.debug("Starting evolution process")
                pop = opt_alg.evolve(pop)

            #log_stats(pop)




def log_stats(pop):
    '''
    Logs statistics of a population
    TODO: Clean up funciton and 
    '''
    fits, vectors = pop.get_f(), pop.get_x()
    mean_cd = []

    ndf, dl, dc, ndr  = pyg.fast_non_dominated_sorting(fits)

    logger.info("Non-dominated front")
    for nd in ndf:
        logger.info("Vector: " + str(vectors[nd]) +"\Fitness: " + str(fits[nd]))
    logger.debug("DL: " + str(dl) + "\nDC: " + str(dc) +"\nNDR: " + str(ndr))

    cr_dist = pyg.crowding_distance(pop.get_f()[ndf])

    logger.info("Means of fitness-vectors: " + str(np.mean(pop.get_f(), axis=0)))
    mean_cd.append(np.mean(cr_dist[np.isfinite(cr_dist)]))
    logger.info("Crowding distance mean and std: %(mean)f +/- %(std)f" %{"mean":np.nanmean(mean_cd), "std":np.nanstd(mean_cd)}) 


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

    args = parser.parse_args()

    # Start sweetspot search
    sweetspot_search(args.codec, args.moga)
