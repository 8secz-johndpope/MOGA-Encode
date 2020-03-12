# By: Oscar Andersson 2019

from flask import Flask, abort, request
import json
from test import main as run_eval
import sys

import time

start_time = time.time()
last_request = time.time()
eval_calls = 0

app = Flask(__name__)

@app.route('/')
def index():
    time_since = str(int(round(time.time() - last_request)))
    mean_call_time = 0
    if eval_calls > 0:
        mean_call_time = round((time.time()-start_time) / eval_calls)
    return ("Time since last request: " + time_since +
            " seconds<br>Number of eval-calls: " + str(eval_calls) +
            "<br>Mean call time: " + str(mean_call_time) + " seconds")


command_args = []
command_args.append("--cfg")
command_args.append("experiments/cityscapes/seg_hrnet_ocr_w48_train_oscar.yaml")
command_args.append("DATASET.ROOT")
command_args.append("/data/HRNet-mldata/")
command_args.append("TEST.MODEL_FILE")
command_args.append("models/hrnet_ocr_cs_8162_torch11.pth")


@app.route('/eval', methods=['GET'])
def get_eval_results():
    global last_request, eval_calls
    print("Running evaluation")

    last_request = time.time()
    eval_calls += 1
    eval_list = request.args.get('eval_list')
    command = command_args
    command.append("DATASET.TEST_SET")
    if(eval_list != None):
        command.append("list/cityscape_eval/"+ str(eval_list) +".lst")
    else:
        command.append("list/cityscapes/val.lst")
    
    eval_results = run_eval(command)
    
    try:
        res = app.response_class(
            status=200,
            response=json.dumps(eval_results),
            mimetype='application/json'
        )
        return res
    except Exception as exception:
        print(exception)
        abort(500)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
