# By: Oscar Andersson 2019

# Imports
import pygmo as pyg
import numpy as np
import random, logging, os, argparse, csv, pickle
from datetime import datetime

# Import modules
import config.config as cfg
import utils.plotting as pl
from optimization_problem import sweetspot_problem


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
        hb = high_bounds[param]
        lb = low_bounds[param]
        if cfg.opt_type[param] != "f":
            hb += 0.4999
            lb -= 0.4999
        stepsize = (hb-lb) / (cfg.POP_SIZE-1)
        
        param_vals = []
        for i in range(0, cfg.POP_SIZE):
            param_vals.append(lb + stepsize*i)
        random.shuffle(param_vals)

        # Round param values if paramtype isnt float
        if cfg.opt_type[param] != "f":
            param_vals = np.around(param_vals, decimals=0)
            param_vals = [0 if val == 0 else val for val in param_vals] # Remove negative zeroes
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

        for epoch in range(1, cfg.EPOCHS+1):

            for rate_control in cfg.RATE_CONTROLS[codec]:
                # Load parameters for codec
                cfg.load_params_from_json(codec, rate_control)
                #logger.info("Optimising " + codec + " using " + rate_control + "...")

                for alg in cfg.MOG_ALGS:
                    cfg.epoch = epoch
                    cfg.mog_alg = alg
                    logger.info("----------- EPOCH: " + str(epoch) + " Alg: " + alg +
                                " Codec: " + str(codec) +":"+str(rate_control)+" ------------")

                    # Get optimization problem and algorithm
                    opt_prob = pyg.problem(sweetspot_problem())

                    # Initiate population
                    rand_seed = epoch*3+1  # An arbitrary but repeatable randomisation seed
                    random.seed(rand_seed)
                    pop = pyg.population(prob=opt_prob, seed=rand_seed)
                    pop = uniform_init(opt_prob, pop)

                    # Set up optimization algorithm
                    opt_alg = get_optimization_algorithm(rand_seed)

                    # Evolve pop using opt_alg
                    logger.debug("Starting evolution process")
                    for gen in range(0, cfg.NO_GENERATIONS):
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


if(__name__ == "__main__"):
    '''
    The main function which starts the entire optimisation algorithm
    '''

    # Create timestamp used for logging and results
    cfg.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cfg.configure_logging()
    logger = logging.getLogger('gen-alg')

    parser = argparse.ArgumentParser(description='Multi objective genetic algorithm')
    parser.add_argument('-a', '--moga', default=None, help="Evaluate using specific ML-algorithm")
    parser.add_argument('-c', '--codec', default=None, help="Evaluate a specific codec")
    parser.add_argument('-rc', '--ratecontrol', default=None, help="Evaluate a specific rate control")
    parser.add_argument('-r', '--resume', action="store_true", help="Resume optimisation using pickle-file")
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
