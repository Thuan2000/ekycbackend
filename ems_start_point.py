'''
Decide with ems service will run when container start
'''
import argparse
import os

from config import *
from core.cv_utils import create_if_not_exist
from utils.logger import logger

if __name__ == '__main__':
    # Main steps
    service_type = os.environ.get('SERVICE_TYPE', SERVICE_DEFAULT)
    print('Service type', service_type)

    print('Run ekyc server')

    from ems import retention_server_face
    server = retention_server_face.RetentionServer()
    server.run()
