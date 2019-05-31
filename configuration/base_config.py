"""
Base configuration for cv-server
"""
import os
import utils.gpu
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.path.pardir))
ROOT = os.path.abspath(os.path.join(parent_dir, os.path.pardir))


class Mode:
    '''
    Mode
    '''
    LOG = True
    QA_MODE = False
    SHOW_FRAME = True
    DEBUG = False
    PRODUCTION = os.environ.get('PRODUCTION', True)
    QUERY_TOP10_MODE = False
    SEND_QUEUE_TO_DASHBOARD = True
    CALC_FPS = True


class Preprocess:
    NMS_MAX_OVERLAP = 1.0
    BGS_RATIO = 0.1 # RATIO BETWEEN BGS BOUNDING BOXES AND ROI


class Model:
    FACENET_DIR = '%s/models/model-20180227-062200.pb' % ROOT
    COEFF_DIR = '%s/models/coefficient_resnet_v1.pb' % ROOT
    MTCNN_DIR = '/home/thuan/models/align'
    # GLASSES_DIR = '%s/models/model_SunGlasses_or_Not.pth' % ROOT
    GLASSES_DIR = '%s/models/resnet18_glasses_and_maskes_p4_finetuned_v3.pth' % ROOT
    MTCNN_MXNET_DIR = '%s/models/mtcnn-model' % ROOT
    MASK_DIR = '%s/models/model_GauzeMasks_or_Not.pth' % ROOT
    HAAR_FRONTAL = '%s/models/haarcascade_frontalface_default.xml' % ROOT
    MARS_DIR = '%s/models/mars-small128.pb' % ROOT
    YOLO_DIR = '%s/models/yolo.h5' % ROOT
    YOLO_DIR_MXNET = '%s/models/yolo3_darknet53_voc_0070_0.8997_topview_1702.params' % ROOT
    ANCHORS_DIR = '%s/models/yolo_anchors.txt' % ROOT
    COCO_CLASSES_DIR = '%s/models/coco_classes.txt' % ROOT
    ARCFACE_DIR = '/home/thuan/models/arcface/model-0000.params'
    MASAN_CUSTOMER_DIR = '%s/models/Model_masan_visitor_security.pth' % ROOT
    GENDER_AGE_DIR = '%s//models/gamodel-r50/model,0' % ROOT
    ANTISPOOFING_3DCNN_DIR = '%s/models/antispoofing-3dcnn-model.hdf5' % ROOT
    ANTISPOOFING_CNN_DIR = '%s/models/epoch_93_loss_0.003309512510895729_acc_0.9991295614328758' % ROOT


class GPU:
    GPU_DEVICE = '/gpu:0'
    GPU_FRACTION = 0.2


class Frame:
    '''
    Configuration for reading frame
    '''
    FRAME_INTERVAL = 1  # read 1 every x frames
    SCALE_FACCTOR = 1  # rescale frame for faster face detection
    SHOULD_CROP = False
    ROI_CROP = (0.5, 0.5, 0.5, 0.5)
    FRAME_QUEUE_SIZE = 200
    STREAM_TIMEOUT = 10
    FRAME_ON_DISK = '%s/data/frame_queue_buffer/' % ROOT
    MAX_FRAME_ON_DISK = 1000000
    # ROI_CROP = (0.35, 0.4, 0.15, 0.1) #TCH


class MTCNN:
    '''
    Configuration for mtcnn face detector
    '''
    MIN_FACE_SIZE = 112  # minimum size of face
    THRESHOLD = [0.6, 0.7, 0.8]  # three steps's threshold
    FACTOR = 0.709  # default 0.709 image pyramid -- magic number
    SCALE_FACTOR = 1
    MIN_FACE_WIDTH = 32
    MIN_FACE_HEIGHT = 32
    # SCALE_FACTOR = 2 for tensorflow mtcnn detector
    NUM_WORKER = 10
    ACCURATE_LANDMARK = False



class Filters:
    YAW = 60
    PITCH = 60
    COEFF = 0.14
    YOLO_SCORE = 0.9
    MIN_PEDESTRIAN_AREA_SIZE = 200**2  # minimum size of pedestrian



class Align:
    '''
    Configuration to align faces
    '''
    MARGIN = 32
    IMAGE_SIZE = 160
    ROI_RATE_W = 110
    ROI_RATE_H = 140
    UPPER_RATE = 0.8
    LOWER_RATE = 1.2
    ASPECT_RATIO = 1.5
    MAX_FACE_ANGLE = 50


class Matcher:
    '''
    Configuration for finding matched id via kd-tree
    '''
    RETRIEVAL_MODEL = 'faiss'
    CLOSE_SET_SVM = False
    MAX_TOP_MATCHES = 99
    INDEXING_INTERVAL = 30 * 60  # how often the matcher will update itself, in seconds
    INDEX_LEAF_SIZE = 2
    EMB_LENGTH = 512
    SVM_PROBS = 0.8
    MIN_ASSIGN_THRESHOLD = 1.1
    MIN_ASSIGN_RATE = 0.5
    NON_FACE = 'NON_FACE'
    NEW_FACE = 'NEW_FACE'
    EXCESS_NEW_REGISTER = 150
    CLEAR_SESSION = False
    ADD_NEW_FACE = True
    USE_IMAGE_ID = False
    HNSW_NROF_THREADS = 2
    HNSW_MAX_ELEMENTS = 1000000
    TRACKER_RADIUS = 1.0
    TOP_MATCHES = 7
    EF = 200


class Track:
    '''
    Configuration for tracking
    '''
    INIT_FACE_ID = 'DETECTING'
    RECOGNIZED_FACE_ID = 'RECOGNIZED'
    BAD_TRACK = 'BAD-TRACK'
    TRACKING_VIDEO_OUT = False
    RECOG_TIMER = 30
    MAX_IOU_DISTANCE = 0.7
    MIN_NOF_TRACKED_FACES = 4
    MIN_NOF_TRACKED_PEDESTRIANS = 40
    RETRACK_MINRATE = 0.6
    VALID_ELEMENT_MINRATE = 0.8
    COMBINED_MATCHER_RATE = 0.7
    HISTORY_RETRACK_MINRATE = 0.3
    UNIQUE_ID_MINRATE = 0.8
    MAX_NROF_ELEMENTS = 100
    SKIP_FRAME = 10
    VIDEO_OUT_PATH = '/mnt/data/TCH_data'
    BB_SIZE = 0
    NUM_IMAGE_PER_EXTRACT = 5
    TRACK_TIMEOUT = 20
    MAX_COSINE_DISTANCE = 0.3
    NN_BUDGET = 40
    USE_DIRECTION = False
    FRAME_STEP = 5
    THRESHOLD_DETECT = 0.4


class Rabbit:
    '''
    Configuration for sending message via rabbitmq
    '''
    USERNAME = 'admin'
    PASSWORD = 'admin123'
    IP_ADDRESS = '210.211.119.152'
    PORT = 5672
    INTER_SEP = '|'
    INTRA_SEP = '?'


class MongoDB:
    '''
    Configuration for MongoDB
    '''
    PORT = 27017
    # USERNAME = 'developer'
    # PASSWORD = 'CodingIsAnArt_x2BGKXxnT4sPNsNdmMQaQ32R'
    # IP_ADDRESS = '210.211.119.152'
    USERNAME = ''
    PASSWORD = ''
    IP_ADDRESS = 'localhost'
    DB_NAME = 'ngoclam1'
    DASHINFO_COLS_NAME = 'dashinfo'
    FACEINFO_COLS_NAME = 'faceinfo'
    PEDESTRIANINFO_COLS_NAME = 'retention_image_pedestrian'
    MSLOG_COLS_NAME = 'mslog'
    TIME_DB_NAME = 'NAS_timestamp'
    INFO = 'dashboard_pedestrian_info'


class RPCServer:
    DST_PATH = r'/home/vtdc/eyeq-card-face-mapping-api'
    SAVE_PATH = r'public/results'


class Worker:
    TASK_TRACKER = 'tracker'
    TASK_DASHBOARD = 'dashboard'
    TASK_EXTRACTION = 'extraction'
    TASK_SAVE_IMAGES = 'save_images'


class Dir:
    DATA_DIR = '%s/data' % ROOT
    TRACKING_DIR = os.path.join(DATA_DIR, 'tracking')
    ANNOTATION_DIR = os.path.join(DATA_DIR, 'annotation')
    DATASET_DIR = os.path.join(DATA_DIR, 'dataset')
    LOG_DIR = os.path.join(DATA_DIR, 'log')

    VIDEO_DIR = os.path.join(DATA_DIR, 'video')
    MATCHER_DIR = os.path.join(DATA_DIR, 'matcher')
    SEND_RBMQ_DIR = os.path.join(DATA_DIR, 'send_rbmq')

    # TODO: Clean up this
    DB_IMAGES_DIR = os.path.join(DATA_DIR, 'dashboard_images')
    SESSION_DIR = '%s/session' % ROOT
    # db contains the main register data, kdtree as pkl10
    DB_DIR = os.path.join(SESSION_DIR, 'db')
    # live save original frame as format AREA_TIME_BB.pkl
    LIVE_DIR = os.path.join(SESSION_DIR, 'live')
    # reg contain register image for each  person
    REG_DIR = os.path.join(SESSION_DIR, 'reg')
    # frames cotains all frame needed for qa
    FRAME_DIR = os.path.join(SESSION_DIR, 'frames')


class PickleFile:
    # TODO: Clean up this
    # a data structure for registered [image_id: face_id]
    REG_IMAGE_FACE_DICT_FILE = os.path.join(Dir.DB_DIR, 'reg_image_face_dict.pkl')
    # a data structure to log changes
    FACE_CHANGES_FILE = os.path.join(Dir.DB_DIR, 'face_changes_list.pkl')
    # a data structure for trained kdtree (timestamp, tree, len, image_ids)
    MATCHER_TUP_FILE = os.path.join(Dir.DB_DIR, 'matcher_tup.pkl')

    # a data structure for trained svm (timestamp, svm, len)
    KDTREE_MATCHER_TUP_FILE = os.path.join(Dir.DB_DIR, 'kdtree_matcher_tup.pkl')
    FAISS_MATCHER_TUP_FILE = os.path.join(Dir.DB_DIR, 'faiss_matcher_tup.pkl')
    # a data structure for traiend kd tree (timestamp, tree, len, image_id)
    SVM_MATCHER_TUP_FILE = os.path.join(Dir.DB_DIR, 'svm_matcher_tup.pkl')
    # a data structure for traiend kd tree (last_time, embs, length)
    LINEAR_MATCHER_TUP_FILE = os.path.join(Dir.DB_DIR, 'linear_matcher_tup.pkl')
    # store the predict id of detected face from live run
    LIVE_DICT_FILE = os.path.join(Dir.DB_DIR, 'live_dict.pkl')
    # keep info for each frame as 22
    FRAME_IMAGE_DICT_FILE = os.path.join(Dir.DB_DIR, 'frames.pkl')
    # run QA mode to generate this to compare against groundtruth, live_dict.pkl
    PREDICTION_DICT_FILE = os.path.join(Dir.DB_DIR, 'prediction.pkl')


class LogFile:
    LOG_NAME = 'debug'
    LOG_FILE = os.path.join(Dir.LOG_DIR, '%s.log' % LOG_NAME)


class Log:
    __default_name = str(round(time.time() * 1000))
    LOG_NAME = os.environ.get('LOG_FILE_NAME', __default_name)
    LOG_FILE = os.path.join(Dir.LOG_DIR, '%s.log' % LOG_NAME)


class MicroServices:
    IMAGES = 'images[]'
    SERVICE_MASAN_CUSTOMER_CLASSIFICATION = 'http://service_massan_customer_classification:9001/classify'
    SERVICE_AGE_GENDER_PREDICTION = 'http://age_gender_prediction:9001/gender-age'
    SERVICE_GLASSES_MASK_CLASSIFICATION = 'http://glasses_mask_classification:9001/glasses-mask-classification'


class GRPCServices:
    DEFAULT_PORT = 50005


class GRPCEmbsExtractor:
    SERVICE_ARCFACE_FEATURE_EXTRACTION_S1 = 'ArcFace_feature_extraction_container_s1:%s' % GRPCServices.DEFAULT_PORT
    SERVICE_ARCFACE_FEATURE_EXTRACTION_S2 = 'ArcFace_feature_extraction_container_s2:%s' % GRPCServices.DEFAULT_PORT
    SERVICE_ARCFACE_FEATURE_EXTRACTION_S3 = 'ArcFace_feature_extraction_container_s3:%s' % GRPCServices.DEFAULT_PORT
    SERVICE_LOAD_BALANCING = 'embs_extract_load_balancer:%s' % GRPCServices.DEFAULT_PORT


class GRPCCoeffExtractor:
    SERVICE_COEFF_EXTRACTION_S1 = 'coeff_extraction_container_s1:%s' % GRPCServices.DEFAULT_PORT
    SERVICE_COEFF_EXTRACTION_S2 = 'coeff_extraction_container_s2:%s' % GRPCServices.DEFAULT_PORT


class MatchingServer:
    SERVER_1 = 'matching_server_container:%s' % GRPCServices.DEFAULT_PORT


class AntiSpoofingServer:
    SERVER_1 = 'anti_spoofing_server_container:%s' % GRPCServices.DEFAULT_PORT


class DetectionServer:
    SERVER_1 = 'detection_server_container:%s' % GRPCServices.DEFAULT_PORT


class FaceExtractionServer:
    SERVER_1 = 'face_extraction_server_container:%s' % GRPCServices.DEFAULT_PORT


class CoeffExtractionServer:
    SERVER_1 = 'coeff_extraction_server_container:%s' % GRPCServices.DEFAULT_PORT
