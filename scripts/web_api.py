#!flask/bin/python
import flask
from gevent.pywsgi import WSGIServer
import argparse
import logging
import sys
import os
from config import config
from common.utils import Utils
from common.model.runner import Runner
from common.dataset.lc_quad import LC_QuAD
from scripts.config_args import parse_args

app = flask.Flask(__name__)
runner = None


@app.route('/link', methods=['POST'])
def link():
    if not flask.request.json:
        flask.abort(400)

    question = flask.request.json['question']
    k = 5
    if 'k' in flask.request.json:
        k = flask.request.json['k']

    try:
        result = 'test'
        if runner is not None:
            result = runner.link(question, k=k, e=0.1)
        return flask.jsonify(result), 201
    except RuntimeError as expt:
        logger.error(expt)
        return flask.jsonify({'error': str(expt)}), 408
    except Exception as expt:
        logger.error(expt)
        return flask.jsonify({'error': str(expt)}), 422


@app.errorhandler(404)
def not_found(error):
    return flask.make_response(flask.jsonify({'error': 'Command Not found'}), 404)


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    Utils.setup_logging()
    args = parse_args()

    dataset = LC_QuAD(config['lc_quad']['train'], config['lc_quad']['test'], config['lc_quad']['vocab'],
                      False, args.remove_stop_words)

    runner = Runner(dataset, args)
    runner.load_checkpoint()
    runner.environment.entity_linker = None
    runner.environment.relation_linker = None

    print(runner.link("Who has been married to both Penny Lancaster and Alana Stewart?", k=10, e=0.1))
    logger.info("Starting the HTTP server")
    http_server = WSGIServer(('', args.port), app)
    http_server.serve_forever()
