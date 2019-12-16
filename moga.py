# By: Oscar Andersson 2019

# Imports
import pygmo as pyg
import numpy as np
import random, logging, os
from datetime import datetime

# Import modules
import config as cfg
import plotting as pl
from optimization_problem import sweetspot_problem

# Global logger object
logger, start_time = None, None


def get_optimization_algorithm(opt_prob, randseed):
    '''
    Returns an optimisation algorithm
    '''
    opt_alg = None
    if(cfg.MOG_ALG == "nsga2"):
        # cr: crossover probability, m: mutation probability
        # eta_c: distribution index for crossover, eta_m: distribution index for mutation
        opt_alg = pyg.algorithm( pyg.nsga2(gen=cfg.NO_GENERATIONS, cr=0.7, m=0.15,
                                           eta_c=10, eta_m=50, seed=randseed) )
        opt_alg.set_verbosity(1)
    elif(cfg.MOG_ALG == "moead"):
        opt_alg = pyg.algorithm ( pyg.moead(gen = cfg.NO_GENERATIONS, weight_generation = "grid",
                                            decomposition = "tchebycheff", neighbours = 5,
                                            CR = 1, F = 0.5, eta_m = 20, realb = 0.9,
                                            limit = 2, preserve_diversity = True) )
    elif(cfg.MOG_ALG == "nspso"):
        opt_alg = pyg.algorithm ( pyg.nspso(gen = cfg.NO_GENERATIONS, omega = 0.6, c1 = 0.01, c2 = 0.5, chi = 0.5,
                                            v_coeff = 0.5, leader_selection_range = 2,
                                            diversity_mechanism = "crowding distance",
                                            memory = False) )
    return opt_alg


def uniform_bitrate_init(prob, pop):
    '''
    Initialize chromosomes of pop with uniform initialization of bitrate gene
    and random initialization for the other genes
    '''
    logger.debug("Populating population...")
    low_bounds, high_bounds = prob.get_bounds()
    stepsize = (high_bounds[0]-low_bounds[0]) // cfg.POP_SIZE
    for i in range(0, cfg.POP_SIZE):
        bitrate = low_bounds[0] + stepsize*i
        x = [bitrate]
        for j in range(1, len(low_bounds)):
            x.append(random.randint(low_bounds[j], high_bounds[j]))
        logger.debug("Pushing chromosome: " + str(x))
        pop.push_back(x)
    return pop



def sweetspot_search():
    '''
    Evolves a population with a given optimization algorithm
    '''

    # Get optimization problem and algorithm
    opt_prob = pyg.problem(sweetspot_problem())

    for i in range(1, cfg.EPOCHS+1):
        cfg.epoch = i
        logger.info("------------------ EPOCH " + str(i) + " -----------------")
        
        # Initiate population
        pop = pyg.population(prob=opt_prob, seed=i*3+1)
        pop = uniform_bitrate_init(opt_prob, pop)
        pl.plot_front_from_pop(pop, "initial population")

        # Set up optimization algorithm
        opt_alg = get_optimization_algorithm(opt_prob, i*3+1)

        # Evolve pop using opt_alg
        logger.debug("Starting evolution process")
        pop = opt_alg.evolve(pop)

        log_stats(pop)




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

    # Start sweetspot search
    sweetspot_search()
