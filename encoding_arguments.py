# By: Oscar Andersson 2019
import logging
logger = logging.getLogger('gen-alg')

def get_codec_args(x, encoder):
    '''
    Sets input and output arguments for encoding according to the
    decision vector "x" and the encoder.
    '''
    # TODO: Check if x can be used directly
    x_new = []
    for x_i in x:
        x_new.append(str(int(x_i)))

    # Set the constant arguments 
    input_args = {
        "pattern_type": 'glob',
        "framerate": "30"
    }
    output_args = {
        "c:v": encoder,
        'b:v':  x_new[0] + "M",
        'pass': "1",
        "format": 'mp4',
        'pix_fmt': 'yuv420p' # ?????? TODO: Look over
    }

    # Set the conditional arguments
    if(encoder == "h264_nvenc"): return get_h264_nvenc_args(input_args, output_args, x_new)
    elif(encoder == "hevc_nvenc"): return get_hevc_nvenc_args(input_args, output_args, x_new)
    elif(encoder == "libx264"): return get_libx264_args(input_args, output_args, x_new)
    elif(encoder == "h264_vaapi"): return get_h264_vaapi_args(input_args, output_args, x_new)
    elif(encoder == "vp9_vaapi"): return get_vp9_vaapi_args(input_args, output_args, x_new)
    elif(encoder == "libvpx-vp9"): return get_libvpxvp9_args(input_args, output_args, x_new)
    elif(encoder == "libaomav1"): return get_libaomav1_args(input_args, output_args, x_new)
    else: raise Exception("Could not find codec")


# TODO: fix
def get_libx264_args(input_args, output_args, x):
    speed_switcher = {"1": "slow", "2": "medium", "3": "fast"}
    profile_switcher = {"1": "baseline", "2": "main", "3": "high"}
    tune_switcher = {"1": "none", "2": "stillimage", "3": "film", "4": "grain"}


    output_args["maxrate"] = x[0]+"M"
    output_args["minrate"] = x[0]+"M"
    output_args["preset"] = speed_switcher[x[1]]
    output_args["profile"] = profile_switcher[x[2]]
    if(tune_switcher[x[3]] != "none"): output_args["tune"] = tune_switcher[x[3]]
    return input_args, output_args

def get_h264_nvenc_args(input_args, output_args, x):
    speed_switcher = {"1": "slow", "2": "medium", "3": "fast"}
    profile_switcher = {"1": "baseline", "2": "main", "3": "high"}
    tune_switcher = {"1": "none", "2": "stillimage", "3": "film", "4": "grain"}

    output_args["preset"] = speed_switcher[x[1]]
    output_args["profile"] = profile_switcher[x[2]]
    if(tune_switcher[x[3]] != "none"): output_args["tune"] = tune_switcher[x[3]]
    return input_args, output_args

def get_hevc_nvenc_args(input_args, output_args, x):
    speed_switcher = {"1": "slow", "2": "medium", "3": "fast"}
    profile_switcher = {"1": "baseline", "2": "main", "3": "high"}

    output_args["preset"] = speed_switcher[x[1]]
    output_args["profile"] = profile_switcher[x[2]]
    return input_args, output_args

# TODO: fix
def get_h264_vaapi_args(input_args, output_args, x):
    speed_switcher = {"1": "slow", "2": "medium", "3": "fast"}
    profile_switcher = {"1": "baseline", "2": "main", "3": "high"}

    input_args = apply_vaapi_input_args(input_args)
    output_args["preset"] = speed_switcher[x[1]]
    output_args["profile"] = profile_switcher[x[2]]
    return input_args, output_args

# TODO: fix
def get_libvpxvp9_args(input_args, output_args, x):
    speed_switcher = {"1": "best", "2": "good", "3": "realtime"}

    output_args["deadline"] = speed_switcher[x[1]]
    return input_args, output_args

# TODO: fix
def get_vp9_vaapi_args(input_args, output_args, x):
    speed_switcher = {"1": "slow", "2": "medium", "3": "fast"}
    profile_switcher = {"1": "baseline", "2": "main", "3": "high"}

    input_args = apply_vaapi_input_args(input_args)
    output_args["preset"] = speed_switcher[x[1]]
    output_args["profile"] = profile_switcher[x[2]]
    return input_args, output_args

# TODO: fix
def get_libaomav1_args(input_args, output_args, x):
    output_args["rom-mt"] = "1"
    output_args["tiles"] = "2x2"
    return input_args, output_args

def apply_vaapi_input_args(input_args):
    input_args["hwaccel"] = "vaapi"
    input_args["hwaccel_output_format"] = "vaapi"
    input_args["hwaccel_device"] = "/dev/dri/renderD128"
    return input_args