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
        self.data_length = None

class Node:
    def __init__(self):
        rospy.init_node('Node')
        self.features = []
        self.total_count = 0
        self.node_id = rospy.get_param(rospy.get_name())
        self.publish_rate = rospy.get_param('publish_rate')
        self.publish_queue_size = rospy.get_param('publish_queue_size')

    def callback(self, feature):
        data, data_length = np.array(feature.data), feature.data_length
        from_id, to_id = feature.header.frame_id, feature.to_id
        data_reshaped = data.reshape(-1, data_length)
        print '\n IN CALLBACK \n'
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
        method = 'SIFT' #Can use 'SIFT','ORB', or 'SURF'

        #Choose file path for main images
        dirname = path.abspath(path.join(__file__, "../.."))
        images_file_path = dirname + '/test_images/*.%s'

        #Build main data structure
        features = Empty_struc()

        # Get raw image files based on node_id
        image_list = []             # features handled by this node
        image_list_other = []       # features handled by other nodes
        for filepaths in [sorted(glob.glob(images_file_path % ext)) for ext in ["jpg", "gif", "png", "tga"]]:
            for filepath in filepaths:
                image_num = int(filepath[57:-4])
                if  image_num == 2 * self.node_id or image_num == 2 * self.node_id + 1:
                    image_list.append(cv2.imread(filepath))
                else:
                    image_list_other.append(cv2.imread(filepath))

        ################### Extract features for images handled by other nodes ###################

        print '\nExtracting features for images handled by other nodes'

        #Import Data into [dxn] numpy array
        features = self.get_imgs_data(features, method, image_list_other)
        features.num_img = len(features.images)

        #This is the length and dimension of the feature vector
        features.size = features.data.shape
        print('Number and Size of Features')
        print(features.size)

        # Flatten the features to 1d vectors
        data_flattened = features.data.flatten()
        print 'data_flattened shape'
        print data_flattened.shape

        # publish the flattened features
        feature.data = data_flattened
        feature.data_length = features.data_length
        pub.publish(feature)

        ################### REPEAT for images handled by this node ###################

        print '\nExtracting features for images handled by this node'

        #Import Data into [dxn] numpy array
        features = self.get_imgs_data(features, method, image_list)
        features.num_img = len(features.images)

        #This is the length and dimension of the feature vector
        features.size = features.data.shape
        print('Number and Size of Features')
        print(features.size)

        data_flattened = features.data.flatten()
        print 'data_flattened shape'
        print data_flattened.shape


    # Import car images and run Sift to get dataset
    def get_imgs_data(self, features, method, image_list):

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

        # original code to get all images in the folder
        # image_list = [cv2.imread(item) for i in [sorted(glob.glob(path % ext)) for ext in ["jpg", "gif", "png", "tga"]] for item in i]

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

        # print 'INSIDE FUNCTION:\n datashape, keypoints shape, member shape:'
        # print features.data.shape
        # print features.keypoints.shape
        # print features.member.shape

        return features

if __name__ == '__main__':
    n = Node()
    n.main()