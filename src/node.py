#! /usr/bin/python

import time
import rospy
import random
import sys
import cv2
import glob

from std_msgs.msg import Int64
from dist_num.msg import Feature # pylint: disable = import-error

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os.path as path

class Empty_struc:
    def __init__(self):
        self.images = None
        self.data = None

class Node:
    def __init__(self):
        rospy.init_node('Node')
        self.collection = [] # list of lists, each list inside is a point/vector
        self.total_count = 0
        self.node_id = rospy.get_param(rospy.get_name())
        self.publish_rate = rospy.get_param('publish_rate')
        self.publish_queue_size = rospy.get_param('publish_queue_size')

    def callback(self, feature):
        data, data_length = feature.data, feature.data_length
        from_id, to_id = feature.header.frame_id, feature.to_id
        data_reshaped = data.reshape(-1, data_length)
        print data_reshaped.shape

    def main(self):
        sub = rospy.Subscriber('/features', Feature, self.callback)
        pub = rospy.Publisher('/features', Feature, queue_size = self.publish_queue_size)

        rate = rospy.Rate(self.publish_rate)
        feature = Feature()

        # pausing for 1 second before start can ensure the first messages are not lost
        # otherwise the first messages are lost
        # number of lost messages = number of nodes initialized
        time.sleep(1)

        #Initialization of Important Constants and Features data structure
        threshold = .75
        method = 'SIFT' #Can use 'SIFT','ORB', or 'SURF'

        #Choose file path for main images
        dirname = path.abspath(path.join(__file__, "../.."))
        images_file_path = dirname + '/test_images/*.%s'

        #Build main data structure
        features = Empty_struc()

        #Import Data into [dxn] numpy array
        features = self.get_imgs_data(features, method, images_file_path)
        features.num_img = len(features.images)

        #This is the length and dimension of the feature vector
        features.size = features.data.shape
        print('Number and Size of Features')
        print(features.size)

        data_flattened = features.data.flatten()
        print data_flattened.shape

        '''
        CALL BACK FINISHED, START PUBLISHING MESSAGES!
        '''

    # Import car images and run Sift to get dataset
    def get_imgs_data(self, features, method, path):

        # Initalize data structures
        if method == 'SIFT':
            features.data = np.empty((1,128), dtype = np.float64)
            features.data_length = 128
            sift = cv2.xfeatures2d.SIFT_create(500, 3, 0.1, 5, 1.6)
        if method == 'ORB':
            features.data = np.empty((1,32), dtype = np.float64)
            features.data_length = 32
            orb = cv2.ORB_create()
        if method == 'SURF':
            features.data = np.empty((1,64), dtype = np.float64)
            features.data_length = 64
            surf = cv2.xfeatures2d.SURF_create(400)

        first = 0
        features.keypoints = np.empty((1,2), dtype = object)
        features.member = np.empty((1), dtype = int)

        # Get raw image files
        image_list = [cv2.imread(item) for i in [sorted(glob.glob(path % ext)) for ext in ["jpg", "gif", "png", "tga"]] for item in i]

        features.images = image_list

        # For each image, extract sift features and organize them into right structures
        for i in range(0, len(features.images)):
            gray= cv2.cvtColor(features.images[i], cv2.COLOR_BGR2GRAY)
            if method == 'SURF':
                kp, des = surf.detectAndCompute(gray, None)
            if method == 'SIFT':
                kp, des = sift.detectAndCompute(gray, None)
            if method == 'ORB':
                kp, des = orb.detectAndCompute(gray, None)
            keypts = [p.pt for p in kp]
            member = np.ones(len(keypts)) * i
            # add features to data structures
            if first > 0:
                features.kpts = np.hstack((features.kpts, kp))
                features.desc = np.vstack((features.desc, des))
                features.keypoints = np.concatenate((features.keypoints, keypts), axis=0)
                features.data = np.concatenate((features.data, des), axis=0)
                features.member = np.concatenate((features.member, member), axis=0)
            if first == 0:
                features.kpts = kp
                features.desc = des
                features.keypoints = np.concatenate((features.keypoints, keypts), axis=0)
                features.data = np.concatenate((features.data, des), axis=0)
                features.member = np.concatenate((features.member, member), axis=0)
                first = 1

        # Delete first empty artifact from stucture def
        features.keypoints = np.delete(features.keypoints, 0, axis=0)
        features.data = np.delete(features.data, 0, axis=0)
        features.data = np.random.normal(features.data, 0.001)
        features.member = np.delete(features.member, 0)

        return features

if __name__ == '__main__':
    n = Node()
    n.main()