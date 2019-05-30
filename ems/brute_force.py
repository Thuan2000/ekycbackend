import os 
import cv2 
from core import face_detector, tf_graph, preprocess, face_extractor
from core.cv_utils import CropperUtils
from config import Config
from core.tracking import detection
from core.face_align import AlignCustom
import numpy as np


class BruteForce():
    def __init__(self):
        face_graph_detector = tf_graph.FaceGraph()
        facenet_graph = tf_graph.FaceGraph()
        self.preprocessor = preprocess.Preprocessor(preprocess.align_and_crop)
        self.aligner = AlignCustom()
        self.prewhitening = False
        try: 
            self.face_detector = face_detector.MtcnnDetectorMxnet()
            self.embs_extractor = face_extractor.ArcFaceExtractor(model_path=Config.Model.ARCFACE_DIR)
        except Exception as e: 
            print('ERROR: ')
            print(e)
        super(BruteForce, self).__init__()

    def run_brute_force(self, frame):
        bbs, pts = self.face_detector.detect_face(frame)
        print('What are the bounding boxes? ')
        print(bbs)

        nrof_faces = len(bbs)

        if nrof_faces == 0: 
            return 'no_face'
        elif nrof_faces != 1:
            return 'multiple_faces'
        else:
            preprocessed_images = []
            for i in range(nrof_faces):
                # cropped_face = CropperUtils.crop_face(frame, bbs[i][:-1])
                preprocessed = self.preprocessor.process(frame, pts[:, i], self.aligner, Config.Align.IMAGE_SIZE, self.prewhitening)

                import cv2
                import time

                print('SaveeeeDDDD!!!!')
                print(preprocessed.shape)
                cv2.imwrite('/home/thuan/test/%s.jpg' % time.time(), preprocessed)
                preprocessed_images.append(preprocessed)
            preprocessed_images = np.array(preprocessed_images)
            if preprocessed_images.any():
                embs = self.embs_extractor.extract_features_all_at_once(preprocessed_images)
                return embs
