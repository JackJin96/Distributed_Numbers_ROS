#!/usr/bin/env python

import rospy
import time
import numpy as np
import sys
import cv2
import glob

from std_msgs.msg import String
from dist_num.msg import Feature, Features # pylint: disable = import-error
from scipy.spatial import distance as scipydist
import matplotlib.pyplot as plt;
import os.path as path

class QuickmatchNode:
    def __init__(self):
        rospy.init_node('QuickmatchNode')
        # Collect feature if necessary, not recommended for current version
        # self.collected_features = []
        # self.collected_feature_members = []
        self.node_id = rospy.get_param("/node_ids" + rospy.get_name())
        self.label_matrix = np.array([])
        self.threshold = 0.75
        self.image_list = self.get_all_images()

    def kmeans_callback(self, msg):
        data = msg.data
        data = np.reshape(data, (-1, 128))
        self.label_matrix = data

    def callback(self, msg):
        data, data_belongs = np.array(msg.data), np.array(msg.data_belongs)
        data_kpts = np.array(msg.data_kpts)

        from_id, to_id = int(msg.header.frame_id), msg.to_id
        if to_id == self.node_id:
            data = np.reshape(data, (-1, 128))
            data_kpts = np.reshape(data_kpts, (-1, 3))
            data_kpts = [cv2.KeyPoint(x, y, size) for x, y, size in data_kpts]
            data_len = len(data)
            data_belongs_len = len(data_belongs)

            while len(self.label_matrix) == 0:
                time.sleep(0.1)
            # Calculate distance matrix
            D = self.distance(data)

            # Calculate Feature Density
            density, bandwidth = self.calc_density(D, data_belongs, len(data))

            # Build Tree
            parent, parent_edge = self.build_kdtree(density, D, len(D))

            # sort the parent edges from shortest to longest and returns the sorted indicies
            sorted_idx = self.sort_edge_index(parent_edge)

            clusters, cluster_member, matchden = self.break_merge_tree(parent, parent_edge, data_belongs, 
                                                                       sorted_idx, bandwidth, self.threshold, 
                                                                       np.shape(density), self.node_id, data_len)

            query_idx = 0
            print('Press Any Key to Show Next Image Set')
            for i in range(0, len(self.image_list)):
                print('Image Index Comparison')
                dmatch = self.features_to_DMatch(data_belongs, data_kpts, cluster_member, D, query_idx, i)
                print '##### dmatch shape: #####'
                print len(dmatch)

    def main(self):
        sub = rospy.Subscriber('/proc_features', Features, self.callback)
        self.sub_kmeans = rospy.Subscriber('/labels', Feature, self.kmeans_callback)

        rospy.spin()

    # Convert sets of images to DMatch stucture
    def features_to_DMatch(self, data_belongs, data_kpts, cluster_member, dist, im_idx1, im_idx2):
        x = np.take(cluster_member, np.where(data_belongs == im_idx1))
        y = np.take(cluster_member, np.where(data_belongs == im_idx2))
        match = np.intersect1d(x,y)
        first = 0
        image1 = self.image_list[im_idx1].copy()
        image2 = self.image_list[im_idx2].copy()

        if match.shape[0] == 0:
            print('No Matches between those images')
            print(im_idx1,im_idx2)
            return()
        for i in range(0, match.shape[0]):
            fet1_idxa = np.where(cluster_member == match[i])
            fet1_idxb = np.where(data_belongs == im_idx1)
            fet1_idx = np.intersect1d(fet1_idxa, fet1_idxb)
            fet2_idxa = np.where(cluster_member == match[i])
            fet2_idxb = np.where(data_belongs == im_idx2)
            fet2_idx = np.intersect1d(fet2_idxa, fet2_idxb)
            desc_dist = dist[fet1_idx, fet2_idx]
            dpoint = cv2.DMatch(fet1_idx, fet2_idx, im_idx1, desc_dist)
            if first == 0:
                DMatches = dpoint
                first = 1
            if first > 0:
                DMatches = np.hstack((DMatches, dpoint))
        print(im_idx1, im_idx2)
        image1 = self.image_list[im_idx1].copy()
        image2 = self.image_list[im_idx2].copy()
        img3 = cv2.drawMatches(image1, data_kpts, image2, data_kpts, DMatches,1)

        cv2.imshow("Image", img3)
        cv2.waitKey(0)
        cv2.destroyWindow("Image")
        return (DMatches)

    def get_all_images(self):
        # Choose file path for main images
        dirname = path.abspath(path.join(__file__, "../.."))
        images_file_path = dirname + '/test_images/*.%s'
        res = []    # images handled by the current node
        for filepaths in [sorted(glob.glob(images_file_path % ext)) for ext in ["jpg", "gif", "png", "tga"]]:
            for filepath in filepaths:
                res.append(cv2.imread(filepath))
        return res

    def break_merge_tree(self, parent, parent_edge, member, sorted_idx, bandwidth, 
                         threshold, size, agent_index=1, max_feats=1):
        # offset = np.multiply(agent_index, max_feats)
        # cluster_member = np.add(np.arange(size[0]), offset)

        cluster_member = np.arange(size[0])
        matchden = bandwidth

        for j in range(0,size[0]):
            idx = sorted_idx[j]
            parent_idx = parent[idx][0]

            if parent_idx != -1:
                min_dens = np.minimum(matchden[idx], matchden[parent_idx])
                x = np.take(member, np.where(cluster_member == cluster_member[parent_idx]))
                y = np.take(member, np.where(cluster_member == cluster_member[idx]))
                isin_truth = np.isin(x, y)

                #Only consider points that meet criteria
                if (parent_edge[idx] < (threshold * min_dens)) and not(isin_truth.any()):
                    cluster_member[cluster_member == cluster_member[idx]] = cluster_member[parent_idx]
                    matchden[cluster_member == cluster_member[idx]] = min_dens
                    matchden[cluster_member == cluster_member[parent_idx]] = min_dens

            (values, counts) = np.unique(cluster_member, return_counts=True)
            clusters = counts
        
        return clusters, cluster_member, matchden

    def sort_edge_index(self, parent_edge):
        sorted_idx = sorted(range(len(parent_edge)), key=lambda k: parent_edge[k])
        return sorted_idx

    def distance(self, points):
        D = scipydist.pdist(points,'euclidean')
        D = scipydist.squareform(D)
        return D

    def build_kdtree(self, density, dist, size):
        #Initialize tree array or arrays
        parent = np.empty(size, dtype=object)
        parent_edge = np.empty(size, dtype=object)

        #Build Tree starts here
        for i in range(0, size):
            parent[i] = np.array([-1])
            parent_edge[i] = np.array([-1])
        #Find larger density nodes here
            larger = np.transpose(np.nonzero(np.greater(density, density[i])))
        #If the node is not the highest, find its parent
            if larger.shape[0] != 0:
                x = np.take(dist[i,:], larger)
                nearest = np.take(larger, (np.where(x == x.min())))
                dist_min = x.min()
                parent[i] = nearest[0]
                parent_edge[i] = dist_min
        return parent, parent_edge

    def calc_density(self, D, member, num_img):
        I = np.identity(D.shape[0]).astype(int)
        Dint = D.astype(int) - I
        D[Dint == -1] = np.nan
        num_feat = D.shape[0]
        bandwidth = np.ones(num_feat)
        # membership = np.asarray(member)
        membership = member.astype(int)
        x = np.empty(num_img,dtype=object)

        #Loop through num images to find normalization for each one
        for i in range(0, num_img):
            x[i] = np.where(membership == i)
        for k in range(0, num_feat):
            first = x[membership[k]][0][0]
            last_idx = x[membership[k]][0].shape[0]
            last = x[membership[k]][0][last_idx-1]
            bandwidth[k] = np.nanmin(D[k,first:last])
        density = np.zeros(num_feat)
        D[np.isnan(D)] = 0

        #Broadcast division of bandwidth
        D_corrected = D/bandwidth[:, None]

        #Calc Gaussian at each feature
        gaus = np.exp((-.5) * np.power(D_corrected, 2))

        #Sum density values at each point
        density = np.sum(gaus, axis=0)

        return density, bandwidth

if __name__ == '__main__':
    qmn = QuickmatchNode()
    qmn.main()