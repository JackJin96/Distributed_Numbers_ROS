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
            # print '\ndata_len: %d, data_belongs_len: %d\n' % (data_len, data_belongs_len)

            while len(self.label_matrix) == 0:
                time.sleep(0.1)

            D = self.distance(data)
            density, bandwidth = self.calc_density(D, data_belongs, len(data))

            print 'NODE ' + str(self.node_id)
            print density.shape
            print bandwidth.shape

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

    def distance(self, points):
        D = scipydist.pdist(points,'euclidean')
        D = scipydist.squareform(D)
        return D

    def process_data(self):
        pass

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
    qmn = QuickmatchNode()
    qmn.main()