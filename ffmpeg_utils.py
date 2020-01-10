# By: Oscar Andersson 2019

import os, logging, ffmpeg
import numpy as np
from encoding_arguments import get_codec_args
import config as cfg
logger = logging.getLogger('gen-alg')


def img_to_vid(images_dir, decision_vector, encoder):
    '''
    Converts images of a directory to a video-file
    '''

    input_args, output_args, is_two_pass = get_codec_args(decision_vector, encoder)

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
            exit(1)

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
        exit(1)
    return True

def vid_to_vid(img_dir, decision_vector, encoder):
    '''
    Converts images of a directory to a video-file
    '''

    _, output_args, is_two_pass = get_codec_args(decision_vector, encoder)

    logger.debug("Vid -> Vid")
    logger.debug(str(output_args))
    in_path = img_dir + "/ll.mkv"
    out_path = cfg.TEMP_STORAGE_PATH

    if(is_two_pass):
        try:
            logger.debug("Applying two passes!")
            output_args["pass"] = "1"
            (
                ffmpeg
                .input(in_path)
                .output("/dev/null", **output_args)
                .global_args('-loglevel', 'error', "-stats", "-hide_banner", "-y")
                .run(capture_stderr=True)
            )
            output_args["pass"] = "2"
        except ffmpeg.Error as ex:
            logger.critical("FFMPEG: error converting images to video")
            logger.critical(ex.stderr.decode('utf8'))
            exit(1)

    try:
        (
            ffmpeg
            .input(in_path)
            .output(out_path, **output_args)
            .global_args('-loglevel', 'error', "-stats", "-hide_banner", "-y")
            .run(capture_stderr=True)
        )
    except ffmpeg.Error as ex:
        logger.critical("FFMPEG: error converting images to video")
        logger.critical(ex.stderr.decode('utf8'))
        exit(1)
    return True


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
            .output(filenaming, compression_level=cfg.IMG_COMP_LVL)  # PNG: comp_lvl 1 seems to be fast yet small enough
            .global_args('-loglevel', 'error', "-stats", "-hide_banner", "-y")
            .run(capture_stderr=True)
        )
    except ffmpeg.Error as ex:
        logger.critical("FFMPEG: error converting video to images")
        logger.critical(ex.stderr.decode('utf8'))
        exit(1)


def transcode(img_path, output_dir, decision_vector, encoder="NA"):
    '''
    Handles the process of transcoding images -> compressed-video -> images
    with potential compression artifacts.
    The function returns the file size of the compressed-video
    '''
    if(encoder == "NA"): encoder = cfg.video_encoder

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    filenames = get_names(img_path)

    img_to_vid(img_path, decision_vector, encoder)
    #vid_to_vid(img_path, decision_vector, encoder)
    vid_to_img(output_dir)

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
    i = 0
    try:
        for i in range(len(generated_names)):  # index 0 -> len(generated_names)-1
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




def loss_test():
    '''
    For debugging
    Test for checking that png compression is completelly lossless.
    '''

    original_image = "data/frankfurt/frankfurt_000000_000294_leftImg8bit.png"

    print("md5 checksum for original image")
    (
        ffmpeg
        .input(original_image)
        .output("-", format="md5")
        .global_args('-loglevel', 'error', "-stats", "-hide_banner", "-nostats")
        .run()
    )

    print("md5 checksums for png-compressed images:  (comp-lvl: 100 and comp-lvl: 0)")
    (
        ffmpeg
        .input(original_image)
        .output("output_img100.png", compression_level=100)
        .global_args('-loglevel', 'error', "-stats", "-hide_banner", "-nostats")
        .run()
    )
    (
        ffmpeg
        .input(original_image)
        .output("output_img0.png", compression_level=0)
        .global_args('-loglevel', 'error', "-stats", "-hide_banner", "-nostats")
        .run()
    )
    (
        ffmpeg
        .input("output_img100.png")
        .output("-", format="md5")
        .global_args('-loglevel', 'error', "-stats", "-hide_banner", "-nostats")
        .run()
    )
    (
        ffmpeg
        .input("output_img0.png")
        .output("-", format="md5")
        .global_args('-loglevel', 'error', "-stats", "-hide_banner", "-nostats")
        .run()
    )



if(__name__ == "__main__"):
    loss_test()


