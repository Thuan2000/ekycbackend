from flask import Flask, request, jsonify
from flask_cors import CORS


class AbstractServer():

    def __init__(self):
        app = Flask(__name__)
        CORS(app)

        @app.after_request
        def after_request(response):
            # response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
            response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
            return response

        self.app = app
        self.app.register_error_handler(Exception, self.handle_server_error)
        self.request = request
        self.add_endpoint()
        self.init()

    def init(self):
        # subclass do building pipeline, ... in here
        pass

    def add_endpoint(self):
        # subclass add api endpoint in this function
        pass

    def run(self):
        self.app.run(host='0.0.0.0', port=9001)

    @staticmethod
    def response_success(result, **kwargs):
        message = kwargs.get('message')
        if message is not None:
            return jsonify({'status': "successful", 'data': result, 'message': message})
        return jsonify({'status': "successful", 'data': result})

    @staticmethod
    def response_error(message, **kwargs):
        return_dict = {'message': message, 'status': 'failed'}
        return_dict.update(kwargs)
        return jsonify(return_dict)

    def handle_server_error(self, e):
        print(e)
        return jsonify({'status': 'failed', 'message': 'Server is not available, Please try again later'})
