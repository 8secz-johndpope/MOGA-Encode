# By: Oscar Andersson 2019

from flask import Flask, abort
import json
from test import main as run_eval
import sys

import time

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello!"


command_args = []
command_args.append("--cfg")
command_args.append("experiments/cityscapes/seg_hrnet_ocr_w48_train_oscar.yaml")
command_args.append("DATASET.ROOT")
command_args.append("/data/")
command_args.append("DATASET.TEST_SET")
command_args.append("list/cityscapes/val_tiny.lst")
command_args.append("TEST.MODEL_FILE")
command_args.append("models/hrnet_ocr_cs_8162_torch11.pth")


@app.route('/eval', methods=['GET'])
def get_eval_results():
    print("Running evaluation")

    eval_results = run_eval(command_args)
    
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
    app.run(host='localhost', port=5001, debug=False)
