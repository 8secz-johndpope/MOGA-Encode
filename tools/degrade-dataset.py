# By: Oscar Andersson 2019

# Imports
import numpy as np
from skimage import io, util, filters, transform
from multiprocessing import Pool
from tqdm import trange
import random, logging, os, argparse, csv, re, json, shutil, time
from datetime import datetime

import Automold as am
import Helpers as hp
import glob
import cv2 as cv2

# Values which regulate data degredation
# Noise parameters
GAUSSIAN_BLUR = 0
GAUSSIAN_NOISE = 0.00075

# Moving
MOVING_INTENSITY = 50
TRANSFORM_MODE = "reflect"


# Variable set in main loop 
DEGRADATION = None



def degrade_dataset(ds_in_path, ds_out_path, degradation):
    global DEGRADATION
    print("Degrading " + ds_in_path)
    DEGRADATION = degradation
    # Check that output dir is present
    if not os.path.exists(ds_out_path):
        os.mkdir(ds_out_path)

    # Use pool of workers to evaluate images in parallel
    pool = Pool(processes=16)
    calc_map = []

    for dir_path in os.listdir(ds_in_path):
        situation_in_path = ds_in_path + "/" + dir_path
        situation_out_path = ds_out_path + "/" + dir_path
        if os.path.isdir(situation_in_path):
            if not os.path.exists(situation_out_path):
                os.mkdir(situation_out_path)
        calc_map.append([
            situation_in_path,
            situation_out_path ])

    pool.map(degrade, calc_map)
    pool.terminate()


def degrade(args):
    if DEGRADATION == "rain":
        rain_and_save(args)
    elif DEGRADATION == "noise":
        add_noise_to_image(args)
    elif DEGRADATION == "moving":
        move_and_save(args)
    else:
        raise Exception("Invalid degradation")


def rain_and_save(args):
    '''Apply rain degradation to a situation'''
    situation_in_path, situation_out_path = args

    image_list = []
    file_out_paths = []

    for f in sorted(os.listdir(situation_in_path)):
        item_local_in_path = situation_in_path + "/" + f
        item_local_out_path = situation_out_path + "/" + f
        if os.path.isfile(item_local_in_path):
            if f.endswith('.png'):
                image = cv2.cvtColor(cv2.imread(item_local_in_path),cv2.COLOR_BGR2RGB)
                rainy_image= am.add_rain(image, slant=-1,drop_length=8,drop_width=1,drop_color=(180,180,190))
                cv2.imwrite(item_local_out_path, cv2.cvtColor(rainy_image, cv2.COLOR_RGB2BGR))


def add_noise_to_image(args):
    '''Apply noise degradation to a situation'''
    situation_in_path, situation_out_path = args

    image_list = []
    file_out_paths = []

    for f in sorted(os.listdir(situation_in_path)):
        item_local_in_path = situation_in_path + "/" + f
        item_local_out_path = situation_out_path + "/" + f
        if os.path.isfile(item_local_in_path):
            if f.endswith('.png'):
                image = io.imread(item_local_in_path)

                # Add noise to image
                if GAUSSIAN_NOISE != 0:
                    image = util.random_noise(image, mode='gaussian', var=GAUSSIAN_NOISE)
                if GAUSSIAN_BLUR != 0:
                    image = filters.gaussian(image, sigma=GAUSSIAN_BLUR, multichannel=True)
                image = util.img_as_ubyte(image)

                # Save image
                io.imsave(item_local_out_path, image)



def move_and_save(args):
    '''Apply movement degradation to a situation'''
    situation_in_path, situation_out_path = args

    direction = (0,0)
    while direction == (0,0):
        direction = (random.randint(-MOVING_INTENSITY, MOVING_INTENSITY), random.randint(-MOVING_INTENSITY, MOVING_INTENSITY))

    img_seq_index = 0

    for f in sorted(os.listdir(situation_in_path)):

        item_local_in_path = situation_in_path + "/" + f
        item_local_out_path = situation_out_path + "/" + f
        if os.path.isfile(item_local_in_path):
            if f.endswith('.png'):
                image = cv2.cvtColor(cv2.imread(item_local_in_path),cv2.COLOR_BGR2RGB)

                img_seq_index += 1
                if img_seq_index != 20:
                    image = transform.warp(image, get_transform(img_seq_index, direction), mode=TRANSFORM_MODE)
                    image = util.img_as_ubyte(image)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(item_local_out_path, image)
                else:
                    shutil.copyfile(item_local_in_path, item_local_out_path)

                if img_seq_index == 30: img_seq_index = 0


def get_transform(img_seq_index, direction):
    '''Return the movement transform for a frame'''
    mult = img_seq_index - 20
    coords = (direction[0] * mult, direction[1] * mult)
    return transform.AffineTransform(translation=coords)


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
            if f.endswith('.'+ 'png'):
                files_under_dir.append(item_local_path)
        elif os.path.isdir(item_full_path):
            files, dirs = get_files_under_dir(top_dir, item_local_path)
            files_under_dir.extend(files)
            dirs_under_dir.append(item_local_path)
            dirs_under_dir.extend(dirs)

    return files_under_dir, dirs_under_dir


if(__name__ == "__main__"):
    '''
    The main function that starts the entire degredation
    '''

    parser = argparse.ArgumentParser(description='Degrade dataset')
    parser.add_argument('-p', '--dataset_path', default=None, help="CSV-file with coding parameters to evaluate")
    parser.add_argument('-d', '--degradation', default="noise", help="The degradation of data: noise, rain, moving")

    args = parser.parse_args()
    if(args.dataset_path == None):
        print("Set path!")
        exit(1)

    path = args.dataset_path
    if(path[-1] == '/'):
        path = path[:-1]


    degrade_dataset(path, path + "_degraded", args.degradation)

