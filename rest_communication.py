import requests, json, logging
import config as cfg
logger = logging.getLogger('gen-alg')


# TODO: This function is tailored for HRNet, changes to it may be needed for other ML systems
def get_eval_from_ml_alg():
    '''
    Sends an evaluation request to the machine learning system.
    The ML system will use the data stored in ML_DATA_OUTPUT as validation set.

    Raises
    ------
    Exception
        If any code other than 200 is in the response

    Returns
    -------
    ML-performance measurement
    '''
    logger.info("Requesting evaluation from ML-algorithm...")
    response = requests.get(cfg.REQUEST_ADDRESS, timeout=120*60)

    if(response.status_code != 200):
        raise Exception('Error encountered while communicating with the ML-algorithm')
    
    json_body = response.json()

    return json_body["mean_IoU"]



if(__name__ == "__main__"):  
    # Main function for debugging communication
    eval_res = get_eval_from_ml_alg()
    print(eval_res)