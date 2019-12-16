# By: Oscar Andersson 2019

import ffmpeg_utils # import functions from ffmpeg_utils.py
import pygmo as pyg
import numpy as np
import os, time, random, csv
import rest_communication
import shutil, logging
import config as cfg
import plotting as pl
logger = logging.getLogger('gen-alg')

# Problem class which defines the problem and contain functions for solving it
class sweetspot_problem:
    """
    A class used as a PyGMO User Defined Problem
    Contains information used to define the optimization problem.
    """

    calls = 0
    fitness_dict = {}
    original_img_size = None
    gen = 0
    fitness_of_gen = []

    def __init__(self):
        self.original_img_size = ffmpeg_utils.get_directory_size(cfg.ML_DATA_INPUT)

    def get_name(self):
        return "Sweetspot problem"

    def get_nobj(self):
    # Multi objective optimization (2 objectives)
        return 2

    def fitness(self, x):
        '''
        The MOGA fitness function.
        Evaluates the fitness of chromosome x by retrieving the compression ratio
        and performance numbers from ml-evaluation.
        Returns the fitness values of x.
        '''

        self.calls += 1 # Keep track of amount of fitness-calls
        logger.info("-------------- Fitness-call " + str(self.calls) + " ---------------")
        x_id = str(x)
        logger.debug("Chromosome: " + x_id)

        if x_id in self.fitness_dict:
            logger.info("Using previous fitness of identical input vector")
            self.store_results(x, self.fitness_dict[x_id])
            return self.fitness_dict[x_id]

        # List all clips to degrade
        clips = []
        for clip in os.listdir(cfg.ML_DATA_INPUT):
            if os.path.isdir(os.path.join(cfg.ML_DATA_INPUT, clip)):
                clips.append(clip)

        # Remove any earlier degraded clips
        for clip in clips:
            try:
                output_clip_dir = cfg.ML_DATA_OUTPUT + clip
                shutil.rmtree(output_clip_dir, ignore_errors=False)
            except:
                logger.warning("Directory: " + str(cfg.ML_DATA_OUTPUT + clip) + " not found, moving on...")

        # Parse x into a dict of encoding arguments
        comp_size = 0

        # Apply degredation to every clip
        for clip in clips:
            input_clip_dir = cfg.ML_DATA_INPUT + clip
            output_clip_dir = cfg.ML_DATA_OUTPUT + clip
            logger.debug("Applying degredation to clip: " + input_clip_dir)
            comp_size += ffmpeg_utils.transcode(input_clip_dir, output_clip_dir, x)

        # Retrieve fitness results
        score = float(rest_communication.get_eval_from_ml_alg())    # Get ML-algorithm results 
        comp_ratio = self.original_img_size/comp_size               # Calc compression-ratio
        logger.info("ML-performance: " + str(score) + "\nComp-ratio: " + str(comp_ratio))

        self.fitness_dict[x_id] = [-score, -comp_ratio]
        self.store_results(x, [-score, -comp_ratio])

        # Minimize bitrate, maximize ML-performance
        return [-score, -comp_ratio]  # maximize obj-func -> put a minus sign in front of obj.

    def store_results(self, x, fitness):
        '''
        Stores decision vectors and their fitness.
        Plots the fitness of all chromosomes of every generation.
        '''
        with open(cfg.FITNESS_DATA_PATH + cfg.timestamp + '/data.csv', mode='a') as data_file:
            data_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            data = np.concatenate((fitness, x), axis=None)
            data = np.insert(data, 0, self.calls)
            data_writer.writerow(data)
        self.fitness_of_gen.append(fitness)
        self.gen += 1
        if( (len(self.fitness_of_gen) == cfg.POP_SIZE) and self.gen != 0):
            pl.plot_front_from_fitness(self.fitness_of_gen, "gen "+str(self.gen))
            self.fitness_of_gen = []


    def get_bounds(self):
    # Return the problem's bound-box
        return (cfg.OPT_LOW_BOUNDS, cfg.OPT_HIGH_BOUNDS)     # [Low-bounds], [high-bounds]

    def get_nix(self):
    # Return the no integer dimensions of problem
        return len(cfg.OPT_PARAMS)



