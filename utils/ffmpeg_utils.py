# By: Oscar Andersson 2019

import os, logging, ffmpeg, time
import numpy as np
from utils.enc_arg_parser import get_codec_args
import config.config as cfg
logger = logging.getLogger('gen-alg')


def img_to_vid(images_dir, decision_vector):
    '''
    Converts images of a directory to a video-file
    '''

    input_args, output_args, is_two_pass = get_codec_args(decision_vector)

    logger.debug("Img -> Vid")
    logger.debug(str(input_args)+ " " +str(output_args))
    filenaming = str(images_dir+'/'+'*.'+ cfg.IMAGE_TYPE)
    out_path = cfg.TEMP_STORAGE_PATH

    if(is_two_pass):
        try:
            logger.debug("Applying two passes!")
            output_args["pass"] = "1"
            (
                ffmpeg
                .input(filenaming, **input_args)
                .output("/dev/null", **output_args)
                .global_args('-loglevel', 'error', "-stats", "-hide_banner", "-y")
                .run(capture_stderr=True)
            )
            output_args["pass"] = "2"
        except ffmpeg.Error as ex:
            logger.critical("FFMPEG: error converting images to video")
            logger.critical(ex.stderr.decode('utf8'))
            logger.critical(output_args)
            raise Exception("Failed transcode")

    try:
        (
            ffmpeg
            .input(filenaming, **input_args)
            .output(out_path, **output_args)
            .global_args('-loglevel', 'error', "-stats", "-hide_banner", "-y")
            .run(capture_stderr=True)
        )
    except ffmpeg.Error as ex:
        logger.critical("FFMPEG: error converting images to video")
        logger.critical(ex.stderr.decode('utf8'))
        logger.critical(output_args)
        raise Exception("Failed transcode")


def vid_to_img(images_dir):
    '''
    Converts a video-file to a set of images
    '''
    logger.debug("Vid -> Img")
    filenaming = images_dir + '/' +cfg.NAMING_SCHEME + '.' +  cfg.IMAGE_TYPE
    file_dir = cfg.TEMP_STORAGE_PATH
    logger.debug("Filenaming: " + filenaming + " Comp_lvl: " + str(cfg.IMG_COMP_LVL))

    try:
        ( 
            ffmpeg
            .input(file_dir)
            .output(filenaming, compression_level=cfg.IMG_COMP_LVL)
            .global_args('-loglevel', 'error', "-stats", "-hide_banner", "-y")
            .run(capture_stderr=True)
        )
    except ffmpeg.Error as ex:
        logger.critical("FFMPEG: error converting video to images")
        logger.critical(ex.stderr.decode('utf8'))
        raise Exception("Failed transcode")


def transcode(img_path, output_dir, decision_vector):
    '''
    Handles the process of transcoding images -> compressed-video -> images
    with potential compression artifacts.
    The function returns the file size of the compressed-video
    '''

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    filenames = get_names(img_path)

    tries1, tries2 = 0, 0
    while(True):
        try:
            img_to_vid(img_path, decision_vector)
            break
        except:
            if(tries1 < 3):
                logger.critical("Retrying img to vid conversion")
                tries1 += 1
                time.sleep(240) # Give time for the failed ffmpeg process to terminate
            else:
                logger.critical("Transcode failed, exiting...")
                exit(1)
    while(True):
        try:
            vid_to_img(output_dir)
            break
        except:
            if(tries2 < 3):
                logger.critical("Retrying vid to img conversion")
                tries2 += 1
                time.sleep(240) # Give time for the failed ffmpeg process to terminate
            else:
                logger.critical("Transcode failed, exiting...")
                exit(1)

    vid_size = os.path.getsize(cfg.TEMP_STORAGE_PATH)

    # Remove temporary video-file
    try: 
        os.remove(cfg.TEMP_STORAGE_PATH)
    except Exception:
        logger.critical("Error removing video tempfile")
        exit(1)

    # Rename the new images to their appropriate names
    set_names(filenames, output_dir)
    return vid_size


def get_names(img_path):
    '''
    Get the filenames of all files in img_path
    '''
    files = []
    for f in os.listdir(img_path):
        if os.path.isfile(os.path.join(img_path, f)):
            if f.endswith('.'+ cfg.IMAGE_TYPE):
                files.append(f)
    files.sort()
    return files


def set_names(filenames, img_path):
    '''
    Rename the files in img_path to filenames in filenames
    '''
    generated_names = get_names(img_path) # Getting names of new images
    framediff = len(generated_names) - len(filenames)
    if(framediff != 0): logger.critical("FRAMECOUNT MISSMATCH of "+str(framediff)+" frames: "+str(img_path))

    i = 0
    try:
        for i in range(len(filenames)):  # index 0 -> len(generated_names)-1
            os.rename( os.path.join(img_path, generated_names[i]) , os.path.join(img_path, filenames[i]) )
    except Exception as ex:
        logger.critical(ex)
        logger.critical("Error while naming degraded images")
        exit(1)


def get_directory_size(img_path):
    '''
    Returns the size of img_path directory
    '''
    accumulated_size = 0
    for dirpath, _, filenames in os.walk(img_path):
        for filename in filenames: accumulated_size += os.path.getsize(os.path.join(dirpath, filename))
    return accumulated_size

