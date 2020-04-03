# By: Oscar Andersson 2019

# Imports
import numpy as np
from skimage import io, util, filters
from multiprocessing import Pool
from tqdm import trange
import random, logging, os, argparse, csv, re, json, shutil, time
from datetime import datetime
import config.config as cfg


# Values which regulate data degredation
GAUSSIAN_BLUR = 0
GAUSSIAN_NOISE = 0.00075


def get_files_under_dir(top_dir, dir_path):
    '''
    Recursive function which traverse subdirectories and lists images
    '''
    files_under_dir = []
    dirs_under_dir = []
    for f in os.listdir(top_dir + dir_path):
        item_local_path = dir_path + "/" + f
        item_full_path = top_dir + item_local_path
        if os.path.isfile(item_full_path):
            if f.endswith('.'+ cfg.IMAGE_TYPE):
                files_under_dir.append(item_local_path)
        elif os.path.isdir(item_full_path):
            files, dirs = get_files_under_dir(top_dir, item_local_path)
            files_under_dir.extend(files)
            dirs_under_dir.append(item_local_path)
            dirs_under_dir.extend(dirs)

    return files_under_dir, dirs_under_dir



def make_noisy_dataset(ds_in_path, ds_out_path):

    print("Degrading " + ds_in_path)

    # Check that output dir is present
    if not os.path.exists(ds_out_path):
        os.mkdir(ds_out_path)

    # Get the filenames of all images in the original dataset
    filenames, dirs = get_files_under_dir(ds_in_path, "")

    for d in dirs:
        full_dir = ds_out_path + "/" + d
        if not os.path.exists(full_dir):
            os.mkdir(full_dir)

    # Use pool of workers to evaluate images in parallel
    pool = Pool(processes=16)
    calc_map = []
    for filename in filenames: 
        calc_map.append([
            ds_in_path + "/" + filename,
            ds_out_path + "/" + filename ])
    pool.map(add_noise_to_image, calc_map)
    pool.terminate()


def add_noise_to_image(args):
    orig_path, noisy_path = args
    image = io.imread(orig_path)

    # Add noise to image
    if GAUSSIAN_NOISE != 0:
        image = util.random_noise(image, mode='gaussian', var=GAUSSIAN_NOISE)
    if GAUSSIAN_BLUR != 0:
        image = filters.gaussian(image, sigma=GAUSSIAN_BLUR, multichannel=True)
    image = util.img_as_ubyte(image)

    # Save image
    io.imsave(noisy_path, image)



if(__name__ == "__main__"):
    '''
    The main function which starts the entire degredation
    '''
    # Create timestamp used for logging and results
    cfg.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    parser = argparse.ArgumentParser(description='Degrade dataset')
    parser.add_argument('-p', '--dataset_path', default=None, help="CSV-file with coding parameters to evaluate")
    # TODO: add options for degredation

    args = parser.parse_args()
    if(args.dataset_path != None):
        cfg.ML_DATA_INPUT = args.dataset_path

    if(cfg.ML_DATA_INPUT[-1] == '/'):
        cfg.ML_DATA_INPUT = cfg.ML_DATA_INPUT[:-1]


    make_noisy_dataset(cfg.ML_DATA_INPUT, cfg.ML_DATA_INPUT + "_degraded")

