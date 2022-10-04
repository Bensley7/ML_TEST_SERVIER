import os
import argparse

from flask import Flask, jsonify, request
import torch

from molecular_ml.prediction.mpn_ffc_predict import predict_singe_from_model as predmpn
from molecular_ml.prediction.attention_lstm_predict import predict_singe_from_model as predlstm
from molecular_ml.models.model import MoleculeAttentionLSTM
from molecular_ml.utils.io import load_checkpoint, load_model, read_cfg

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def apicall():
    """API Call
    """
    try:
        smile_json = request.get_json().get("smile")

    except Exception as e:
        raise e

    if not smile_json:
        return(bad_request())
    else:
        res = None
        global model, model_name, label_names
        if model_name == "attention_lstm":
            res = predlstm(model, smile_json, label_names)
        elif model_name == "mpn_ffd":
            res = predmpn(model, smile_json, label_names)

        responses = jsonify(res)
        responses.status_code = 200
        
        return (responses)


@app.errorhandler(400)
def bad_request(error=None):
	message = {
			'status': 400,
			'message': 'Bad Request: ' + request.url + '--> Please check your data payload...',
	}
	resp = jsonify(message)
	resp.status_code = 400

	return resp

def parse_opt():
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument('model_path', type=str, help='weight path of the model')
    parser.add_argument("--config_file", default="", help="path to config file", type=str)

    parser.add_argument("--host", default='0.0.0.0', help="host name", type=str)
    parser.add_argument("--port", default=8000, help="port name", type=int)
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # Parse options
    args = parse_opt()

    #Read config file
    cfg = read_cfg(args.config_file)
    model_name = cfg.MODEL.name
    label_names = cfg.DATA_DIGEST.labels_name

    # Check type of device
    if "cuda" in cfg.MODEL.device:
        print('Moving model to cuda')
    device = torch.device(cfg.MODEL.device)

    # Load model
    model = None
    if cfg.MODEL.name == "attention_lstm":
        model = MoleculeAttentionLSTM()
        model = load_model(model, args.model_path)
    elif cfg.MODEL.name == "mpn_ffd":
        model = load_checkpoint(args.model_path, cfg, device=device)
    model.eval()

    # Run app
    app.run(host=args.host, port=args.port)