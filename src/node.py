#! /usr/bin/python

import time
import rospy
import random
import sys
import cv2
import glob
import yaml
import datetime

from std_msgs.msg import Int64
from dist_num.msg import Feature, Features # pylint: disable = import-error
from scipy.spatial import distance_matrix
from scipy.cluster import vq as clusteralgos

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os.path as path

class Empty_struc:
    def __init__(self):
        self.images = []
        self.data = []
        self.data_length = 0

class Node:
    def __init__(self):
        rospy.init_node('Node')
        self.collected_features = []
        self.collected_feature_members = [] 
        self.total_count = 0
        self.node_id = rospy.get_param("/node_ids" + rospy.get_name())
        if self.node_id == 0:
            self.first_node = True
        else:
            self.first_node = False
        self.publish_rate = rospy.get_param('publish_rate')
        self.publish_queue_size = rospy.get_param('publish_queue_size')
        self.feature_method = rospy.get_param('feature_extration_method')
        self.num_agents = rospy.get_param('num_agents')
        self.label_matrix = []

    def kmeans_callback(self, msg):
        data = msg.data
        data = np.reshape(data, (-1, 128))
        self.label_matrix = data

    def collection_callback(self, msg):
        data = np.array(msg.data)
        from_id, to_id = int(msg.header.frame_id), msg.to_id
        if to_id == self.node_id:
            data = np.reshape(data, (-1, 128))
            data_len = len(data)
            self.total_count += data_len
            for f in data:
                self.collected_features.append(f)
            self.collected_feature_members += [from_id] * data_len
            # cluster_centers = clusteralgos.kmeans2(self.collected_features, 3)
            # print '\n IN collection_callback NODE ' + str(self.node_id) + '\n'
            # print 'DATA recieved! Node' + str(self.node_id) + ' Total recieved: ' + str(self.total_count)
            # print 'DATA recieved! Node' + str(self.node_id) + ' length recieved: ' + str(len(data))
            # print 'Self contains: ' + str(len(self.collected_features))
            # print '\n END collection_callback \n'

    def main(self):

        sub = rospy.Subscriber('/features', Feature, self.collection_callback)
        self.sub_kmeans = rospy.Subscriber('/labels', Feature, self.kmeans_callback)
        pub = rospy.Publisher('/features', Feature, queue_size = self.publish_queue_size)
        self.pub_to_proc = rospy.Publisher('/proc_features', Features, queue_size = self.publish_queue_size)
        self.pub_labels = rospy.Publisher('/labels', Feature, queue_size = self.publish_queue_size)

        # pausing for 1 second before start can ensure the first messages are not lost
        # otherwise the first messages are lost
        # number of lost messages = number of nodes initialized
        time.sleep(1)

        rate = rospy.Rate(self.publish_rate)

        #Initialization of Important Constants and Features data structure
        threshold = .75

        # Build main data structure
        features = Empty_struc()

        # Get raw image files based on node_id
        image_list = self.get_image_paths()             # features handled by the current node

        num_features_extracted = 0
        first_image = True # flag to compute the k-means partitions

        start_time = datetime.datetime.now()

        # For each image
        for image in image_list:
            # temporarily store the features to publish
            publish_store = {}  # to_node_id: [features]
            
            # Import Data into [dxn] numpy array
            features = self.extract_features(features, [image])
            num_features_extracted += len(features.data)

            # Run k-means on the features
            if self.first_node and first_image:
               self.run_and_publish_kmeans(features.data)
               first_image = False
            
            # For the first time running the node, we need to get the labels generated by k-means
            while self.label_matrix == []:
                time.sleep(0.1)

            # Compute distance between each feature to all the feature labels
            D = distance_matrix(features.data, self.label_matrix)
            D = np.square(D) # pylint: disable = assignment-from-no-return

            # Find the minimum distance and corresponding cluster number
            cluster_numbers = np.argmin(D, 1)

            num_col_features = 0
            # For each cluster number in all the cluster numbers
            for i in range(len(cluster_numbers)):
                cur_cluster = cluster_numbers[i]
                # if it matches the label for this node
                if cur_cluster == self.node_id:
                    num_col_features += 1
                    # add it to the node's feature collection
                    self.collected_features.append(features.data[i])
                    self.collected_feature_members.append(self.node_id)
                else: 
                    # if it belongs to other nodes, add it to publish_store
                    publish_store.setdefault(cur_cluster, [])
                    publish_store[cur_cluster].append(features.data[i])
            # tempprint = "node %d collected %d features" % (self.node_id, num_col_features)
            # print tempprint
            
            start_time = self.check_time_publish(start_time)

            num_pub_features = 0
            # Go through publish_store and publish all the features
            for to_node_id, pub_features in publish_store.items():
                num_pub_features += len(pub_features)
                msg = Feature()
                msg.data = np.array(pub_features).flatten()
                msg.to_id = to_node_id
                msg.header.frame_id = str(self.node_id)
                pub.publish(msg)
            # temp = "\nnode %d published %d features\n" % (self.node_id, num_pub_features)
            # print temp
            rate.sleep()
        
        time.sleep(2)

        start_time = self.check_time_publish(start_time)
        print '\nNum of features extracted: ' + str(num_features_extracted) + '\n'
        print 'Node' + str(self.node_id) + ' number of features collected: ' + str(len(self.collected_features))
        print 'Node' + str(self.node_id) + ' number of members collected: ' + str(len(self.collected_feature_members))

    # check if time reaches one second,
    # if so, publish existing features and empty collection
    def check_time_publish(self, start_time):
        end_time = datetime.datetime.now()
        if end_time - start_time >= datetime.timedelta(seconds = 1):
            msg = Features()
            msg.data = np.array(self.collected_features).flatten()
            msg.data_belongs = np.array(self.collected_feature_members)
            msg.header.frame_id = str(self.node_id)
            msg.to_id = self.node_id
            self.collected_features = []
            self.collected_feature_members = []
            start_time = end_time
            self.pub_to_proc.publish(msg)
        return start_time
    
    def get_image_paths(self):
        # Choose file path for main images
        dirname = path.abspath(path.join(__file__, "../.."))
        images_file_path = dirname + '/test_images/*.%s'
        res = []    # images handled by the current node
        for filepaths in [sorted(glob.glob(images_file_path % ext)) for ext in ["jpg", "gif", "png", "tga"]]:
            for filepath in filepaths:
                image_num = int(filepath[57:-4])
                if  image_num == 2 * self.node_id + 1 or image_num == 2 * self.node_id + 2:
                    res.append(cv2.imread(filepath))
        return res
    
    def run_and_publish_kmeans(self, data):
        kmeans_output = clusteralgos.kmeans2(data, self.num_agents)
        center_points = kmeans_output[0]

        print 'CENTER POINTS:\n'
        print center_points.shape
        # print all(center_points)

        # Write the center matrix to the parameter yaml file
        # with open('/home/tron_ubuntu2/catkin_ws/src/dist_num/config/params.yaml','r') as yamlfile:
        #     cur_yaml = yaml.safe_load(yamlfile) # Note the safe_load
        #     cur_yaml['label_matrix'] = center_points.tolist()

        # if cur_yaml:
        #     with open('/home/tron_ubuntu2/catkin_ws/src/dist_num/config/params.yaml','w') as yamlfile:
        #         yaml.safe_dump(cur_yaml, yamlfile) # Also note the safe_dump
        
        center_points_flattened = center_points.flatten()
        msg = Feature()
        msg.data = center_points_flattened
        msg.header.frame_id = str(self.node_id)
        self.pub_labels.publish(msg)

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