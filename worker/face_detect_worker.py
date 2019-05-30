import time
from core import face_detector, tf_graph
from pipe import worker, task
from pipe.trace_back import process_traceback
from utils.logger import logger
from config import Config
from multiprocessing import Value
from ctypes import c_bool
import numpy as np

class MxnetFaceDetectWorker():
    '''
    Detect face from frame stage
    Input: frame, frame_id
    Output: bbs (bouncing boxes), pts (landmarks)
    # 798.5 MiB in GPU Ram
    '''

    def __init__(self, **kwargs):
        self.face_detector = kwargs.get('face_detector')
        super().__init__()

    def doInit(self):
        # try:
        # self.face_detector = face_detector.MtcnnDetectorMxnet()
        # except:
        #     logger.exception("CUDA device out of memory")
        super(MxnetFaceDetectWorker, self).__init__()
        print("MxnetFaceDetectWorker", '=' * 10)

    @process_traceback
    def doFrameTask(self, _task):
        # start = time.time()
        data = _task.depackage()
        frame, frame_info = data['frame'], data['frame_info']

        # timer.detection_start()
        bbs, pts = self.face_detector.detect_face(frame)


        # timer.detection_done()
        # logger.info(
        #     'Frame: %s, bbs: %s, pts: %s' % (frame_info, list(bbs), list(pts)))
        _task = task.Task(task.Task.Face)
        _task.package(bbs=bbs, pts=pts, frame=frame, frame_info=frame_info)
        return _task

class FaceDetectWorker():
    '''
    Detect face from frame stage
    Input: frame, frame_id
    Output: bbs (bouncing boxes), pts (landmarks)
    # 798.5 MiB in GPU Ram
    '''

    def __init__(self, **kwargs):
        self.face_detector = kwargs.get('face_detector')
        super().__init__()

    def doInit(self):
        # try:
        # self.face_detector = face_detector.MtcnnDetectorMxnet()
        # except:
        #     logger.exception("CUDA device out of memory")
        super(FaceDetectWorker, self).__init__()
        print("FaceDetectWorker", '=' * 10)

    @process_traceback
    def doFrameTask(self, _task):
        # start = time.time()
        data = _task.depackage()
        frame, frame_info = data['frame'], data['frame_info']

        # timer.detection_start()
        bbs, pts = self.face_detector.detect_face(frame)

        indices = np.array([i for i, bb in enumerate(bbs) if (bb[2]-bb[0])*(bb[3]-bb[1])/(frame.shape[0]*frame.shape[1]) >= 0.01])
        if indices.size > 0:
            bbs = bbs[indices]
            pts = pts[:, indices]
        else:
            bbs = np.empty((0, 5))
            pts = np.empty((10, 0))

        # timer.detection_done()
        # logger.info(
        #     'Frame: %s, bbs: %s, pts: %s' % (frame_info, list(bbs), list(pts)))
        _task = task.Task(task.Task.Face)
        _task.package(bbs=bbs, pts=pts, frame=frame, frame_info=frame_info)
        return _task

