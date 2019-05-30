from configuration.base_config import *
import configuration.base_config as Base

class MongoDB(Base.MongoDB):
    IP_ADDRESS = 'common_mongo_container'
    DB_NAME = 'checkin-demo-prod'
    FACE_COLS_NAME = 'register_faces'
    USER_COLS_NAME = 'user'
    STATISTICS_COLS_NAME = 'api_statistics'
    USERNAME = ''
    PASSWORD = ''

class LogFile:
    LOG_NAME = os.environ.get('CV_SERVER_NAME', 'demo')
    LOG_FILE = os.path.join(Dir.LOG_DIR, '%s.log' % LOG_NAME)

class Socket:
    HOST = 'https://register.eyeq.tech'
    PORT = 80
    IMAGE_NAMESPACE = '/eyeq-face-recognition'
    IMAGE_EVENT = 'image'
    RESULT_EVENT = 'result'
    MAX_QUEUE_SIZE = 50

class ActionType:
    REGISTER = 'register'
    RECOGNIZE = 'recognize'
    GLASS_MASK_CLASSIFICATION = 'glass_mask_classification'

class Status:
    SUCCESSFUL = 'successful'
    FAIL = 'fail'
    NO_FACES = 'no_faces'


class Matching:
    MATCHING_THRESHOLD = 1.1 #0.66
    POPULARITY = 0.5


class Server:
    FRAME_READER_TIMEOUT = 5
    SOCKET_IMAGE_TIMEOUT = 5
    MAX_IMAGES_LENGTH = 50


class Matcher(Base.Matcher):
    HNSW_MAX_ELEMENTS = 1000000
    HNSW_NROF_THREADS = 1


class MTCNN(Base.MTCNN):
    SCALE_FACTOR = 1

