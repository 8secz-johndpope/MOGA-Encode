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
        'b:v':  str(int(decision_vector[0])) + "M",
        "format": 'mp4',
        'pix_fmt': 'yuv420p' # ?????? TODO: Look over
    }

    for i in range(1, len(cfg.opt_params)):
        vector_val = int(decision_vector[i])
        param_name = cfg.opt_params[i]
        logger.debug("Adding encoding argument:  Param_name: " + param_name +
                     " vectorval: " + cfg.opt_values[param_name][vector_val])
        output_args[param_name] = cfg.opt_values[param_name][vector_val]



    # Set the conditional arguments
    if(encoder == "h264_nvenc"): return get_h264_nvenc_args(input_args, output_args, decision_vector)
    elif(encoder == "hevc_nvenc"): return get_hevc_nvenc_args(input_args, output_args, decision_vector)
    elif(encoder == "libx264"): return get_libx264_args(input_args, output_args, decision_vector)
    elif(encoder == "h264_vaapi"): return get_h264_vaapi_args(input_args, output_args, decision_vector)
    elif(encoder == "hevc_vaapi"): return get_h264_vaapi_args(input_args, output_args, decision_vector)
    elif(encoder == "vp9_vaapi"): return get_vp9_vaapi_args(input_args, output_args, decision_vector)
    elif(encoder == "libvpx-vp9"): return get_libvpxvp9_args(input_args, output_args, decision_vector)
    elif(encoder == "libaomav1"): return get_libaomav1_args(input_args, output_args, decision_vector)
    else: raise Exception("Could not find codec")



def get_libx264_args(input_args, output_args, x):
    output_args["maxrate"] = str(x[0])+"M"
    output_args["minrate"] = str(x[0])+"M"

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
    return input_args, output_args, False

def apply_vaapi_input_args(input_args):
    input_args["hwaccel"] = "vaapi"
    input_args["hwaccel_output_format"] = "vaapi"
    input_args["hwaccel_device"] = "/dev/dri/renderD128"
    return input_args