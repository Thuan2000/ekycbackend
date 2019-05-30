'''
Wrapper around all configuration
To use, first add configuration dir to python path
export PYTHONPATH=
export PYTHONPATH=$(pwd)/configuration:$PYTHONPATH
export PYTHONPATH=$(pwd):$PYTHONPATH
'''
import os

SERVICE_DEFAULT = 'atm_authentication'


service_type = os.environ.get('SERVICE_TYPE', SERVICE_DEFAULT)

import configuration.atm_authentication as Config
print('Use default config')
