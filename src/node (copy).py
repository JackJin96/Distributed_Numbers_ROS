#! /usr/bin/python

import time
import rospy
import random
import sys
import cv2
import glob
import yaml

from std_msgs.msg import Int64
from dist_num.msg import Feature # pylint: disable = import-error
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

    def callback(self, feature):
        data = np.array(feature.data)
        from_id, to_id = int(feature.header.frame_id), feature.to_id
        if to_id == self.node_id:
            data = np.reshape(data, (-1, 128))
            data_len = len(data)
            self.total_count += data_len
            for f in data:
                self.collected_features.append(f)
            self.collected_feature_members += [from_id] * data_len
            # cluster_centers = clusteralgos.kmeans2(self.collected_features, 3)
            # print '\n IN CALLBACK NODE ' + str(self.node_id) + '\n'
            # print 'DATA recieved! Node' + str(self.node_id) + ' Total recieved: ' + str(self.total_count)
            # print 'DATA recieved! Node' + str(self.node_id) + ' length recieved: ' + str(len(data))
            # print 'Self contains: ' + str(len(self.collected_features))
            # print '\n END CALLBACK \n'

    def main(self):

        sub = rospy.Subscriber('/features', Feature, self.callback)
        pub = rospy.Publisher('/features', Feature, queue_size = self.publish_queue_size)

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
        labels_obtained = False

        # For each image
        for image in image_list:
            # temporarily store the features to publish
            publish_store = {}  # to_node_id: [features]
            
            # Import Data into [dxn] numpy array
            features = self.extract_features(features, [image])
            num_features_extracted += len(features.data)

            # Run k-means on the features
            if self.first_node and first_image:
                self.run_and_write_kmeans(features.data)
                first_image = False
            
            # For the first time running the node, we need to get the labels generated by k-means
            if not labels_obtained:
                time.sleep(1)
                self.label_matrix = np.array(rospy.get_param('label_matrix'))
                self.feature_label = self.label_matrix[self.node_id]
                labels_obtained = True
                print self.label_matrix

            # Compute distance between each feature to all the feature labels
            D = distance_matrix(features.data, self.label_matrix)

            # Find the minimum distance and corresponding cluster number
            cluster_numbers = np.argmin(D, 1)

            # For each cluster number in all the cluster numbers
            for i in range(len(cluster_numbers)):
                cur_cluster = cluster_numbers[i]
                # if it matches the label for this node
                if cur_cluster == self.node_id:
                    # add it to the node's feature collection
                    self.collected_features.append(features.data[i])
                    self.collected_feature_members.append(self.node_id)
                else: 
                    # if it belongs to other nodes, add it to publish_store
                    publish_store.setdefault(cur_cluster, [])
                    publish_store[cur_cluster].append(features.data[i])

            # Go through publish_store and publish all the features
            for to_node_id, pub_features in publish_store.items():
                msg = Feature()
                msg.data = np.array(pub_features).flatten()
                msg.to_id = to_node_id
                msg.header.frame_id = str(self.node_id)
                pub.publish(msg)
            rate.sleep()
        print '\nNum of features extracted: ' + str(num_features_extracted) + '\n'
        print 'Node' + str(self.node_id) + ' number of features collected: ' + str(len(self.collected_features))
        print 'Node' + str(self.node_id) + ' number of members collected: ' + str(len(self.collected_feature_members))

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
    
    def run_and_write_kmeans(self, data):
        kmeans_output = clusteralgos.kmeans2(data, self.num_agents)
        center_points = kmeans_output[0]
        # print '\n K means center points'
        # print center_points.shape
        # print type(center_points)
        # print '\n\n'
        rospy.set_param('label_matrix', center_points.tolist())
        # Write the center matrix to the parameter yaml file
        # with open('/home/tron_ubuntu2/catkin_ws/src/dist_num/config/params.yaml','r') as yamlfile:
        #     cur_yaml = yaml.safe_load(yamlfile) # Note the safe_load
        #     cur_yaml['label_matrix'] = center_points.tolist()

        # if cur_yaml:
        #     with open('/home/tron_ubuntu2/catkin_ws/src/dist_num/config/params.yaml','w') as yamlfile:
        #         yaml.safe_dump(cur_yaml, yamlfile) # Also note the safe_dump

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
    
    def cal_density(self, D, member, num_img):
        I = np.identity(D.shape[0]).astype(int)
        Dint = D.astype(int) - I
        D[Dint == -1] = np.nan
        num_feat = D.shape[0]
        bandwidth = np.ones(num_feat)
        membership = np.asarray(member)
        membership = membership.astype(int)
        x = np.empty(num_img,dtype=object)

        #Loop through num images to find normalization for each one
        for i in range(0, num_img):
            x[i] = np.where(membership == i)
        for k in range(0, num_feat):
            first = x[membership[k]][0][0]
            last_idx = len(x[membership[k]][0])
            if last_idx > 1:
                last = x[membership[k]][0][last_idx-1]
                bandwidth[k] = np.nanmin(D[k, first:last+1])
            else:
                bandwidth[k] = 1
        density = np.zeros(num_feat)
        D[np.isnan(D)] = 0

        #Broadcast division of bandwidth
        D_corrected = D/bandwidth[:, None]

        #Calc Gaussian at each feature
        gaus = np.exp((-.5) * np.power(D_corrected, 2))

        #Sum density values at each point
        density = np.sum(gaus,axis=0)

        return density, bandwidth


if __name__ == '__main__':
    n = Node()
    n.main()