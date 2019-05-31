import os
import time

import cv2
import numpy as np

import imageio
from config import Config
from core.cv_utils import create_if_not_exist, base64str_to_frame
from ems import base_server
from pipe import task
from utils import dict_and_list, simple_request, utils
from utils.logger import logger
from utils.utils import get_class_attributes
from worker import (face_detect_worker, face_extract_worker,
                    face_preprocess_worker)
from core import face_detector, face_extractor, tf_graph
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class RetentionServer(base_server.AbstractServer):
    def __init__(self):
        imageio.plugins.freeimage.download()
        self.build_pipeline()
        super().__init__()

    def add_endpoint(self):
        self.app.add_url_rule('/register',
                              'register',
                              self.register_api,
                              methods=['POST'])

    def build_pipeline(self):
        face_graph = tf_graph.FaceGraph()
        mx_detector = face_detector.MTCNNDetector(face_graph, scale_factor=Config.MTCNN.SCALE_FACTOR)
        embs_extractor = face_extractor.ArcFaceExtractor(
            model_path=Config.Model.ARCFACE_DIR)
        self.stageDetectFace = face_detect_worker.FaceDetectWorker(
            face_detector=mx_detector)
        self.stagePreprocess = face_preprocess_worker.PreprocessDetectedArcFaceWorker(
        )
        self.stageExtract = face_extract_worker.MultiArcFacesExtractWorker(
            embs_extractor=embs_extractor)

        self.stageDetectFace.doInit()
        self.stagePreprocess.doInit()
        self.stageExtract.doInit()

    def rotate_image(self, mat, angle):
        """
        Rotates an image (angle in degrees) and expands image to avoid cropping
        """

        height, width = mat.shape[:2] # image shape has 3 dimensions
        image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

        rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

        # rotation calculates the cos and sin, taking absolutes of those.
        abs_cos = abs(rotation_mat[0,0]) 
        abs_sin = abs(rotation_mat[0,1])

        # find the new width and height bounds
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)

        # subtract old image center (bringing image back to origo) and adding the new image center coordinates
        rotation_mat[0, 2] += bound_w/2 - image_center[0]
        rotation_mat[1, 2] += bound_h/2 - image_center[1]

        # rotate image with the new bounds and translated rotation matrix
        rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
        return rotated_mat

    def _get_max_coeff_from_faces(self, _task):
        return max([t.quality for t in _task.depackage()['faces']])

    def run_face_tasks(self, _task):
        rotate_angles = [0, 90, 180, 270]
        tasks = []
        for angle in rotate_angles:

            tmp_task = task.Task(task.Task.Frame)
            tmp_task.package(frame=self.rotate_image(_task.depackage()['frame'], angle),
                             frame_info=_task.depackage()['frame_info'])

            tmp_task = self.stageDetectFace.doFrameTask(tmp_task)
            tmp_task = self.stagePreprocess.doFaceTask(tmp_task)
            tmp_task = self.stageExtract.doFaceTask(tmp_task)
            if len(tmp_task.depackage()['faces']) > 0:
                tasks.append(tmp_task)

        return max(tasks, key=self._get_max_coeff_from_faces)

    def __create_save_dir(self, _time=time.time()):
        save_dir = os.path.join(
            Config.Dir.DATA_DIR,
            '%s_%s' % (utils.timestamp_to_datetime(_time), time.time()))
        save_dir = save_dir.replace(' ', '_')
        create_if_not_exist(save_dir)
        return save_dir

    def save_and_get_images_path(self, receiveTimestamp):
        images = []
        image_paths = []
        save_dir = self.__create_save_dir(receiveTimestamp)
        save_folder = os.path.basename(save_dir)
        if 'video' in self.request.files:
            video_path = os.path.join(save_dir, str(time.time()))
            self.request.files.get('video').save(video_path)
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            while ret:
                image_name = '%s.jpg' % time.time()
                image_path = os.path.join(save_dir, image_name)
                cv2.imwrite(image_path, frame)
                image_paths.append(os.path.join(save_folder, image_name))
                images.append(frame)
                ret, frame = cap.read()

        elif 'images[]' in self.request.files:
            image_files = self.request.files.getlist('images[]')
            for image in image_files:
                image_name = '%s.jpg' % time.time()
                image_path = os.path.join(save_dir, image_name)
                image.save(image_path)
                image_paths.append(os.path.join(save_folder, image_name))
                _image = cv2.imread(image_path)
                if _image is not None:
                    images.append(_image)

        elif 'images[]' in self.request.form:
            images_str = self.request.form.getlist('images[]')
            for image_str in images_str:
                success, frame = base64str_to_frame(image_str)
                if success:
                    image_name = '%s.jpg' % time.time()
                    image_path = os.path.join(save_dir, image_name)
                    cv2.imwrite(image_path, frame)
                    image_paths.append(os.path.join(save_folder, image_name))
                    images.append(frame)
        images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]
        return images, image_paths

    def validate_input_data(self):
        # getting data
        input_data = {}
        input_data['receiveTimestamp'] = round(time.time(), 3)
        return input_data

    def match(self, task_1, task_2):
        face_1 = task_1.depackage()['faces']

        if len(face_1) > 1:
            return 'multiple_faces', 200, {'Content-Type':'text/plain'}
        face_1 = face_1[0].embedding

        face_2 = task_2.depackage()['faces']
        if len(face_2) > 1:
            return 'multiple_faces', 200, {'Content-Type':'text/plain'}
        face_2 = face_2[0].embedding

        dist = np.linalg.norm(face_1 - face_2)
        if dist < 1.0:
            return 'yes', 200, {'Content-Type':'text/plain'}
        else:
            return 'no', 200, {'Content-Type':'text/plain'}

    def recognition(self, input_data, images, image_paths):
        # this client_id should be id to seperate between multi request that send to this same api
        input_date = utils.timestamp_to_datetime(
            input_data['receiveTimestamp'])
        print('\ngot {} images to match, at {}'.format(len(images),
                                                       input_date))

        tasks = []
        if len(images) > 2:
            return 'multiple_faces', 200, {'Content-Type':'text/plain'}
        for image in images[:2]:
            _task = task.Task(task.Task.Frame)
            _task.package(frame=image,
                          frame_info=input_data['receiveTimestamp'])
            # self.recognition_pipeline.put(_task)
            _task = self.run_face_tasks(_task)
            if _task:
                print(_task.depackage())
            tasks.append(_task)

        # notify pipeline there is no more images and start to matching
        try:
            return self.match(tasks[0], tasks[1])
        except Exception as e:
            print(e)
            return 'no_face', 200, {'Content-Type':'text/plain'}

    def register_api(self):
        input_data = self.validate_input_data()
        images, image_paths = self.save_and_get_images_path(
            input_data['receiveTimestamp'])
        return self.recognition(input_data, images, image_paths)
