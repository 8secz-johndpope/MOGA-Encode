# By: Oscar Andersson 2019
import logging
logger = logging.getLogger('gen-alg')
import config as cfg

def get_codec_args(decision_vector, encoder):
    '''
    Sets input and output arguments for encoding according to the
    decision vector "x" and the encoder.
    '''

    # Set the constant arguments 
    input_args = {
        "pattern_type": 'glob',
        "framerate": "30"
    }
    output_args = {
        "c:v": encoder,
        "format": 'mp4',
        'pix_fmt': 'yuv420p' # ?????? TODO: Look over
    }

    for i in range(0, len(cfg.opt_params)):
        arg_val = None
        param_name = cfg.opt_params[i]
        if cfg.opt_type[i] == "c":
            vector_val = int(round(decision_vector[i]))
            arg_val = cfg.opt_cat_values[param_name][vector_val]
        elif cfg.opt_type[i] == "i":
            arg_val = int(round(decision_vector[i]))
        else:
            arg_val = decision_vector[i]
        logger.debug("Adding encoding argument:  Param_name: " + param_name +
        " arg_val: " + str(arg_val))
        output_args[param_name] = arg_val


    if 'b:v' in output_args: output_args['b:v'] += "M"



    # Set the conditional arguments
    if(encoder == "h264_nvenc"): return get_h264_nvenc_args(input_args, output_args, decision_vector)
    elif(encoder == "hevc_nvenc"): return get_hevc_nvenc_args(input_args, output_args, decision_vector)
    elif(encoder == "libx264"): return get_libx264_args(input_args, output_args, decision_vector)
    elif(encoder == "libx265"): return get_libx265_args(input_args, output_args, decision_vector)
    elif(encoder == "h264_vaapi"): return get_h264_vaapi_args(input_args, output_args, decision_vector)
    elif(encoder == "hevc_vaapi"): return get_h264_vaapi_args(input_args, output_args, decision_vector)
    elif(encoder == "vp9_vaapi"): return get_vp9_vaapi_args(input_args, output_args, decision_vector)
    elif(encoder == "libvpx-vp9"): return get_libvpxvp9_args(input_args, output_args, decision_vector)
    elif(encoder == "libaom-av1"): return get_libaomav1_args(input_args, output_args, decision_vector)
    else: raise Exception("Could not find codec")



def get_libx264_args(input_args, output_args, x):
    
    is_two_pass = False

    # Constant bitrate with constrained encoding
    if 'b:v' in output_args: 
        output_args["maxrate"] = output_args['b:v']

        try:
            output_args["bufsize"] = str(round(float(x[0])*float(output_args["bufratio"]), 7)) + "M"
            del output_args["bufratio"]
        except Exception:
            logger.critical("Could not set buffer size, quitting...")
            exit(1)

        # Checks if one or two passes are to be done
        if output_args["pass"]=="2": is_two_pass = True
        del output_args["pass"]

    if(output_args["coder"] == "vlc"): del output_args["trellis"]

    # Remove tune flag if no tuning parameter is passed
    try:
        if(output_args["tune"] == "none"): del output_args["tune"]
    except Exception:
        logger.debug("No tune parameter found, continuing...")

    return input_args, output_args, is_two_pass



def get_libx265_args(input_args, output_args, x):

    # TODO: fix conditions

    if(output_args["coder"] == "vlc"): del output_args["trellis"]

    # Remove tune flag if no tuning parameter is passed
    try:
        if(output_args["tune"] == "none"): del output_args["tune"]
    except Exception:
        logger.debug("No tune parameter found, continuing...")

    # Checks if one or two passes are to be done
    is_two_pass = True if output_args["pass"]=="2" else False
    del output_args["pass"]

    return input_args, output_args, is_two_pass


def get_h264_nvenc_args(input_args, output_args, x):
    # TODO: check if special arguments are to be added

    return input_args, output_args, False


def get_hevc_nvenc_args(input_args, output_args, x):
    # TODO: check if special arguments are to be added
    input_args["hwaccel"] = "nvdec"
    output_args["b_ref_mode:v"]= "middle"

    return input_args, output_args, False


def get_h264_vaapi_args(input_args, output_args, x):
    # TODO: check if more special arguments are to be added
    input_args = apply_vaapi_input_args(input_args)
    return input_args, output_args, False


def get_hevc_vaapi_args(input_args, output_args, x):
    # TODO: check if more special arguments are to be added
    input_args = apply_vaapi_input_args(input_args)
    return input_args, output_args, False


def get_vp9_vaapi_args(input_args, output_args, x):
    # TODO: check if special arguments are to be added
    input_args = apply_vaapi_input_args(input_args)
    return input_args, output_args, False


def get_libvpxvp9_args(input_args, output_args, x):
    # TODO: check if special arguments are to be added

    # Checks if one or two passes are to be done
    is_two_pass = True if output_args["pass"]=="2" else False
    del output_args["pass"]

    output_args["row-mt"] = "1"
    output_args["tiles"] = "2x2"

    return input_args, output_args, is_two_pass


def get_libaomav1_args(input_args, output_args, x):
    # TODO: check if special arguments are to be added
    output_args["row-mt"] = "1"
    output_args["tiles"] = "2x2"
    output_args["threads"] = "16"
    output_args["strict"] = "experimental"
    return input_args, output_args, False

def apply_vaapi_input_args(input_args):
    input_args["hwaccel"] = "vaapi"
    input_args["hwaccel_output_format"] = "vaapi"
    input_args["hwaccel_device"] = "/dev/dri/renderD128"
    return input_args