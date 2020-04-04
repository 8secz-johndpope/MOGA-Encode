# By: Oscar Andersson 2019

import pygmo as pyg
import numpy as np
import os, time, random, csv
import shutil, logging

import config.config as cfg
import utils.ffmpeg_utils as ffu # import functions from ffmpeg_utils.py
import utils.plotting as pl
import utils.rest_communication as rest_com

logger = logging.getLogger('gen-alg')

# Problem class which defines the problem and contain functions for solving it
class sweetspot_problem:
    """
    A class used as a PyGMO User Defined Problem
    Contains information used to define the optimization problem.
    """

    calls = None
    fitness_dict = None
    times = None
    original_img_size = None
    gen = None
    fitness_of_gen = None
    complete_results = None


    def __init__(self):
        self.calls = 0
        self.fitness_dict = {}
        self.times = {}
        self.gen = 0
        self.fitness_of_gen = []
        self.complete_results = {}
        self.original_img_size = ffu.get_directory_size(cfg.ML_DATA_INPUT)
        logger.debug("Problem initiated")


    def get_name(self):
        return "Sweetspot problem"


    def get_nobj(self):
    # Multi objective optimization (2 objectives)
        return 2


    def get_bounds(self):
    # Return the problem's bound-box
        return (cfg.opt_low_bounds, cfg.opt_high_bounds)     # [Low-bounds], [high-bounds]


    def get_nix(self):
    # Return the no integer dimensions of problem
        return len(cfg.opt_params)-cfg.no_continous


    def fitness(self, x):
        '''
        The MOGA fitness function.
        Evaluates the fitness of chromosome x by retrieving the compression ratio
        and performance numbers from ml-evaluation.
        Returns the fitness values of x.
        '''

        self.calls += 1 # Keep track of amount of fitness-calls
        logger.info("------------- Fitness-call " + str(self.calls) + " ---------------")
        x_id = str(np.round(x, 5))
        logger.debug("Chromosome: " + x_id)

        if x_id in self.fitness_dict:
            logger.info("Using previous fitness of identical input vector")
            self.store_results(x, self.fitness_dict[x_id], self.times[x_id], self.complete_results[x_id])
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
        start_time = time.time()
        logger.info("Starting trancode process")
        for clip in clips:
            input_clip_dir = cfg.ML_DATA_INPUT + clip
            output_clip_dir = cfg.ML_DATA_OUTPUT + clip
            logger.debug("Applying degredation to clip: " + input_clip_dir)
            comp_size += ffu.transcode(input_clip_dir, output_clip_dir, x)
        transcode_time = time.time() - start_time
        logger.info("Time for transcode: " + str(int(round(transcode_time)))+" seconds")

        # Retrieve fitness results
        logger.info("Requesting evaluation from ML-algorithm...")
        score, full_response = rest_com.get_eval_from_ml_alg()    # Get ML-algorithm results 
        comp_ratio = self.original_img_size/comp_size               # Calc compression-ratio
        logger.info("ML-performance: " + str(score) + "\nComp-ratio: " + str(comp_ratio))
        logger.debug("All measures: " + str(full_response))

        self.fitness_dict[x_id] = [-score, -comp_ratio]
        self.times[x_id] = transcode_time
        self.complete_results[x_id] = full_response
        self.store_results(x, [-score, -comp_ratio], transcode_time, full_response)

        # Minimize bitrate, maximize ML-performance
        return [-score, -comp_ratio]  # maximize obj-func -> put a minus sign in front of obj.


    def store_results(self, x, fitness, time, full_response):
        '''
        Stores decision vectors and their fitness.
        Plots the fitness of all chromosomes of every generation.
        '''
        self.fitness_of_gen.append(fitness)
        logger.info("Entering store_results!\n Fitness_o_gen: " +
                     str(len(self.fitness_of_gen)) + "  epoch: " + str(cfg.epoch) + 
                     "  decision vector: " + str(x))
        self.update_data_csv(x, fitness, time, full_response)


        if( (len(self.fitness_of_gen) >= cfg.POP_SIZE)):

            # Name used to identify plots and files
            name = (cfg.video_encoder + ":" + cfg.rate_control + " at epoch:" +
                    str(cfg.epoch) + " of MOGA: " + cfg.mog_alg)

            # If this is the last generation, plot and print extended epoch information
            if(self.gen == cfg.NO_GENERATIONS):
                logger.info("Last generation, collecting the fitness of all decision vectors evaluated")
                self.write_ndf_csv(name)

            # Plot generation specific information
            pl.plot_front(name +" at gen:"+str(self.gen), self.fitness_of_gen)
            self.fitness_of_gen = []
            self.gen += 1



    def update_data_csv(self, x, fitness, time, full_response):
        with open(cfg.FITNESS_DATA_PATH + cfg.timestamp + '/data.csv', mode='a') as data_file:
            data_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            data = [str(self.calls), str(cfg.epoch), cfg.video_encoder, cfg.mog_alg, np.array2string(x, precision = 6, separator=','), str(time)]
            data = np.concatenate((data, fitness), axis=None)
            data = np.concatenate((data, str(full_response)), axis=None)
            data_writer.writerow(data)

    def write_ndf_csv(self, name):
        fitness_keys = [ key for key in self.fitness_dict.keys() ]
        fitness_values = [ val for val in self.fitness_dict.values() ]
        _, _, dc, ndr  = pyg.fast_non_dominated_sorting(fitness_values)
        ndf = pyg.non_dominated_front_2d(fitness_values)
        
        logger.info(name + "\nNon dominated vectors: "+str(len(ndf)) + "\nDomination count: " + str(dc) +"\nNon domination ranks: " + str(ndr))
        pl.plot_front(name +" all fits", fitness_values, ndf)
        
        # Save ndf results to file
        with open(cfg.FITNESS_DATA_PATH + cfg.timestamp + '/NDF-' + name + '.csv', mode='w') as data_file:
            data_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for i in ndf:
                data = np.concatenate((fitness_keys[i], fitness_values[i]), axis=None)
                data = np.concatenate((data, self.complete_results[fitness_keys[i]]), axis=None)
                data_writer.writerow(data)