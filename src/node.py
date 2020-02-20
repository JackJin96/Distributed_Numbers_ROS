#! /usr/bin/python

import time
import rospy
import random
import sys
import cv2
import glob

from std_msgs.msg import Int64
from dist_num.msg import Feature # pylint: disable = import-error
from scipy.spatial import distance_matrix

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os.path as path

class Empty_struc:
    def __init__(self):
        self.images = None
        self.data = None
        self.data_length = None

class Node:
    def __init__(self):
        rospy.init_node('Node')
        self.features = []
        self.total_count = 0
        self.node_id = rospy.get_param("/node_ids" + rospy.get_name())
        self.publish_rate = rospy.get_param('publish_rate')
        self.publish_queue_size = rospy.get_param('publish_queue_size')
        self.feature_method = rospy.get_param('feature_extration_method')
        if self.feature_method == 'SIFT':
            self.features = np.random.rand(3, 128) * 255
        elif self.feature_method == 'ORB':
            # check if ORB feature value is from 0 to 32
            self.features = np.random.rand(3, 32) * 32
        elif self.feature_method == 'SURF':
            # check if SURF feature value is from 0 to 32
            self.features = np.random.rand(3, 64) * 32
        else:
            raise Exception('Feature extraction method not supported.')

    def callback(self, feature):
        data, data_length = np.array(feature.data), feature.data_length
        from_id, to_id = feature.header.frame_id, feature.to_id
        if to_id == self.node_id:
            data_reshaped = data.reshape(-1, data_length)
            print '\n IN CALLBACK NODE ' + str(self.node_id) + '\n'
            print 'reshaped data size: '
            print data_reshaped.shape
            print '\n END CALLBACK \n'

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
        self.feature_method = 'SIFT' #Can use 'SIFT','ORB', or 'SURF'

        #Choose file path for main images
        dirname = path.abspath(path.join(__file__, "../.."))
        images_file_path = dirname + '/test_images/*.%s'

        #Build main data structure
        features = Empty_struc()

        # Get raw image files based on node_id
        image_list = []             # features handled by this node
        for filepaths in [sorted(glob.glob(images_file_path % ext)) for ext in ["jpg", "gif", "png", "tga"]]:
            for filepath in filepaths:
                image_num = int(filepath[57:-4])
                if  image_num == 2 * self.node_id + 1 or image_num == 2 * self.node_id + 2:
                    image_list.append(cv2.imread(filepath))

        data_flattened = features.data.flatten()

        # publish the flattened features
        feature.data = data_flattened
        feature.data_length = features.data_length
        feature.to_id = rospy.get_param('/to_ids' + rospy.get_name())
        pub.publish(feature)

        #Import Data into [dxn] numpy array
        features = self.extract_features(features, image_list)
        features.num_img = len(features.images)

        #Compute distance between each feature to all the feature labels

        #Find the minimum distance and corresponding label

        #For each feature that doesn't belong to this node's label,
        #publish the feature to other corresponding node

        #This is the length and dimension of the feature vector
        features.size = features.data.shape
        print '\nNumber and Size of Features (Node ' + str(self.node_id) + ')'
        print features.size
        print '\nmin = ' + str(features.data.min())
        print '\nmax = ' + str(features.data.max())

    # Import car images and run Sift to get dataset
    def extract_features(self, features, image_list):

        # Initalize data structures
        if self.feature_method == 'SIFT':
            features.data = np.empty((1,128), dtype = np.float64)
            features.data_length = 128
            sift = cv2.xfeatures2d.SIFT_create(500, 3, 0.1, 5, 1.6)
        if self.feature_method == 'ORB':
            features.data = np.empty((1,32), dtype = np.float64)
            features.data_length = 32
            orb = cv2.ORB_create()
        if self.feature_method == 'SURF':
            features.data = np.empty((1,64), dtype = np.float64)
            features.data_length = 64
            surf = cv2.xfeatures2d.SURF_create(400)

        first = 0
        features.keypoints = np.empty((1,2), dtype = object)
        features.member = np.empty((1), dtype = int)

        # original code to get all images in the folder
        # image_list = [cv2.imread(item) for i in [sorted(glob.glob(path % ext)) for ext in ["jpg", "gif", "png", "tga"]] for item in i]

        features.images = image_list

        # For each image, extract sift features and organize them into right structures
        for i in range(0, len(features.images)):
            gray= cv2.cvtColor(features.images[i], cv2.COLOR_BGR2GRAY)
            if self.feature_method == 'SURF':
                kp, des = surf.detectAndCompute(gray, None)
            if self.feature_method == 'SIFT':
                kp, des = sift.detectAndCompute(gray, None)
            if self.feature_method == 'ORB':
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