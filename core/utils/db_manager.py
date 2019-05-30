from config import Config
import numpy as np
import time
import os
from multiprocessing import Manager, Process
from utils import database
import threading

def call_inside(Cls):

    class ClassWrapper(object):

        def __init__(self, *args, **kwargs):
            self.oInstance = Cls(*args, **kwargs)

        def __getattribute__(self, s):
            """
            this is called whenever any attribute of a NewCls object is accessed. This function first tries to
            get the attribute off NewCls. If it fails then it tries to fetch the attribute from self.oInstance (an
            instance of the decorated class). If it manages to fetch the attribute from self.oInstance, and
            the attribute is an instance method then `time_this` is applied.
            """
            try:
                x = super(ClassWrapper, self).__getattribute__(s)
            except AttributeError:
                pass
            else:
                return x

            try:
                x = self.oInstance.__getattribute__(s)
            except AttributeError:
                x = self.oInstance.database.__getattribute__(s)
            return x

    return ClassWrapper

@call_inside
class DatabaseManager():
    
    def __init__(self, use_image_id=True):
        self.database = database.RetentionDashboardBatchQueryDatabase()
        manager = Manager()

        self.from_img_id = manager.dict()
        self.from_face_id = manager.dict()

        """
        self.from_img_id : multi process dictionary
            {<imageId>: {'faceId': face_id, \
                'embedding': embedding}}
        self.from_face_id : multi process dictionary
            {<faceId>: {<imageId1>, <imageId2>, ...}
        """
        self.use_image_id = use_image_id

        self.setup()

    def setup(self):
        cursor = self.database.mongodb_faceinfo.find({'isRegistered': False}, \
            projection={'_id': False, 'imageId': True, \
                        'faceId': True, 'embedding': True})

        for document in cursor:
            self.from_img_id[document['imageId']] = \
                {'faceId': document['faceId'], \
                 'embedding': document['embedding']}
            if document['faceId'] not in self.from_face_id.keys():
                 self.from_face_id[document['faceId']] = set()
            self.from_face_id[document['faceId']].add(document['imageId'])

    def get_labels_and_embs(self, face_id=None):
        labels = []
        embs = []
        if self.use_image_id:
            if face_id is None:
                for image_id in self.from_img_id.keys():
                    labels.append(image_id)
                    embs.append(self.from_img_id[image_id]['embedding'])
            else:
                for image_id in self.from_face_id[face_id]:
                    labels.append(image_id)
                    embs.append(self.from_img_id[image_id]['embedding'])
        else:
            return self.database.get_labels_and_embs(face_id=face_id)
        return np.array(labels), np.array(embs).squeeze()

    def find_face_id_by_image_id_in_faceinfo(self, image_ids):
        return [self.from_img_id[image_id]['faceId'] for image_id in image_ids]

    def insert_new_face(self, **args):
        if not args['is_registered']:
            self.from_img_id[args['imageId']] = \
                {'faceId': args['faceId'], \
                 'embedding': args['embedding']}
            if args['faceId'] not in self.from_face_id.keys():
                 self.from_face_id[args['faceId']] = set()
            self.from_face_id[args['faceId']].add(args['imageId'])
        t = threading.Thread(target=(self.database.insert_new_face), \
                kwargs=args)
        t.start()