#!/usr/bin/env python

import rospy
import time
import numpy as np

from std_msgs.msg import String
from dist_num.msg import Feature, Features # pylint: disable = import-error
from scipy.spatial import distance as scipydist
import matplotlib.pyplot as plt;

class QuickmatchNode:
    def __init__(self):
        rospy.init_node('QuickmatchNode')
        # Collect feature if necessary, not recommended for currrent version
        # self.collected_features = []
        # self.collected_feature_members = []
        self.node_id = rospy.get_param("/node_ids" + rospy.get_name())
        self.label_matrix = np.array([])

    def kmeans_callback(self, msg):
        data = msg.data
        data = np.reshape(data, (-1, 128))
        self.label_matrix = data

    def callback(self, msg):
        data, data_belongs = np.array(msg.data), np.array(msg.data_belongs)
        from_id, to_id = int(msg.header.frame_id), msg.to_id
        if to_id == self.node_id:
            data = np.reshape(data, (-1, 128))
            data_len = len(data)
            data_belongs_len = len(data_belongs)
            # DEBUG PRINT
            # print '\ndata_len: %d, data_belongs_len: %d\n' % (data_len, data_belongs_len)

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

            ##### DEBUG PRINT #####
            print 'NODE ' + str(self.node_id)
            print sorted_idx
            # print density
            # print type(bandwidth)
            # print parent
            # print parent_edge

            # Plot bar graph of density if desired
            # y_pos = np.arange(len(density))
            # plt.bar(y_pos, density, align='center', alpha=0.5)
            # plt.show()

    def main(self):
        sub = rospy.Subscriber('/proc_features', Features, self.callback)
        self.sub_kmeans = rospy.Subscriber('/labels', Feature, self.kmeans_callback)

        rospy.spin()

        # #Set rate to use (in Hz)
        # rate = rospy.Rate(1)

        # i = 0
        # #Use sleep to allow subscriber to set up in a period of time
        # rate.sleep()
        # while not rospy.is_shutdown():
        #     #Write to console
        #     rospy.loginfo(msg.data)
        #     #Publish
        #     pub.publish(msg.data)
        #     #Wait until it is done
        #     rate.sleep()

    def break_merge_tree(self, parent, parent_edge, member, sorted_idx, bandwidth, 
                         threshold, size, agent_index=1, max_feats=1):
        offset = np.multiply(agent_index, max_feats)
        cluster_member = np.add(np.arange(size[0]), offset)
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

            (values,counts) = np.unique(cluster_member, return_counts=True)
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
        membership = np.asarray(member)
        membership = membership.astype(int)
        x = np.empty(num_img,dtype=object)

        #Loop through num images to find normalization for each one
        for i in range(0, num_img):
            x[i] = np.where(membership == i)
        for k in range(0, num_feat):
            last_idx = len(x[membership[k]][0])
            if last_idx > 1:
                first = x[membership[k]][0][0]
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
        density = np.sum(gaus, axis=0)

        return density, bandwidth

if __name__ == '__main__':
    qmn = QuickmatchNode()
    qmn.main()