import time
import numpy as np
from core import face_extractor, tf_graph, preprocess
from pipe import worker, task
from pipe.trace_back import process_traceback
from core.cv_utils import CropperUtils
from utils.logger import logger
from config import Config


class MultiArcFacesExtractWorker():
    '''
    Extract features from faces, do multi-faces at a time to reduce time
    Input: faces, preprocessed_images, frame, frame_info
    Output: A single face at a time for tracking
    - face, frame, frame_info
    '''

    def __init__(self, **kwargs):
        self.embs_extractor = kwargs.get('embs_extractor')
        super().__init__()


    # 936.0 MiB
    def doInit(self, use_coeff_filter=True):

        # self.embs_extractor = face_extractor.ArcFaceExtractor(model_path=Config.Model.ARCFACE_DIR)
        self.use_coeff_filter = use_coeff_filter
        if use_coeff_filter:
            coeff_graph = tf_graph.FaceGraph()
            self.coeff_extractor = face_extractor.FacenetExtractor(
                coeff_graph, model_path=Config.Model.COEFF_DIR)
        print("MultiArcFacesExtractWorker", '=' * 10)

    @process_traceback
    def doFaceTask(self, _task):
        # detect all at once, no cuda memory may occur
        data = _task.depackage()
        faces, preprocessed_images, preprocessed_coeff_images, frame, frame_info = data['faces'], data[
            'images'], data['coeff_images'], data['frame'], data['frame_info']
        logger.info("Extract: %s, #Faces: %s" % (frame_info, len(faces)))

        # timer.extract_start()
        # TODO: we can use only one extracter
        face_infos = []
        if preprocessed_images.any():
            embs, _ = self.embs_extractor.extract_features_all_at_once(preprocessed_images)
            coeffs = [100]*embs.shape[0]
            if self.use_coeff_filter:
                _, coeffs = self.coeff_extractor.extract_features_all_at_once(preprocessed_coeff_images)
            # timer.extract_done()
            for i, face in enumerate(faces):
                face.embedding = embs[i, :]
                face.set_face_quality(coeffs[i])
                face_infos.append(face)

        _task = task.Task(task.Task.Face)
        _task.package(faces=face_infos)
        return _task
