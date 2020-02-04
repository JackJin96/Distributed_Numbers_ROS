#! /usr/bin/python

'''
Node 1 handles numbers from 1 to 5
'''

import time
import rospy
import random

from std_msgs.msg import Int64
from dist_num.msg import Feature # pylint: disable = import-error

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

class node0:
    def __init__(self):
        rospy.init_node('node0')
        self.collection = [] # list of lists, each list inside is a point/vector
        self.total_count = 0
        self.node_id = rospy.get_param(rospy.get_name())
        self.matrix = rospy.get_param('defined_matrix')
        # self.label_vector = self.matrix[self.node_id]
        self.publish_rate = rospy.get_param('publish_rate')
        self.publish_queue_size = rospy.get_param('publish_queue_size')
        self.amount_generated = rospy.get_param('amount_generated')
        self.lower_bound = rospy.get_param('lower_bound')
        self.higher_bound = rospy.get_param('higher_bound')

    # Input: two row vectors/lists of the same length
    # Output: Euclidian distance between the two vectors
    def dist_bet_vectors(self, v1, v2):
        return np.sqrt(np.sum((v1[i] - v2[i]) ** 2 for i in range(len(v1))))

    # Input: a randomly generated vector and a given matrix
    # Output: index of the closest row vector in the given matrix
    def closest_vector(self, generated_vector, matrix):
        min_dist = res_index = float('-inf')
        for r in range(len(matrix)):
            dist = self.dist_bet_vectors(generated_vector, matrix[r])
            if dist < min_dist:
                min_dist = dist
                res_index = r
        return res_index

    def callback(self, feature):
        data, from_id, to_id = feature.data, feature.header.frame_id, feature.to_id
        if to_id == self.node_id:
            self.total_count += 1
            self.d[data] = self.d.get(data, 0) + 1

    def main(self):
        sub = rospy.Subscriber('/numbers', Feature, self.callback)
        pub = rospy.Publisher('/numbers', Feature, queue_size = self.publish_queue_size)

        rate = rospy.Rate(self.publish_rate)
        feature = Feature()

        for i in range(self.amount_generated):
            # generate random vector with set higher bound and lower bound
            generated_vector = random.uniform(self.lower_bound, self.higher_bound)
            belongs_to_index = self.closest_vector(generated_vector, self.matrix)

            ############### continue here!!! ######################

            # The integer generated is in the range for the node to process itself
            if belongs_to_index == self.node_id:
                self.total_count += 1
                self.collection.append(generated_vector)
            # otherwise broadcast it in a message through rostopic
            else:
                feature.data = generated_int
                feature.header.frame_id = str(self.node_id)
                feature.to_id = 2
                pub.publish(feature)
            rate.sleep()
        
        print "**********************************************"
        print "NODE 1 FINAL RESULT: "
        print "Total numbers processed: " + str(self.total_count)
        print "(number: frequency)"
        print self.d
        print "**********************************************"

        # Plot bar graph for distribution of frequencies
        objects = [str(k) for k in self.d.keys()]
        y_pos = np.arange(len(objects))
        performance = [v for v in self.d.values()]

        plt.bar(y_pos, performance, align='center', alpha=0.5)
        plt.xticks(y_pos, objects)
        plt.ylabel('Frequency')
        plt.title('Frequency of random numbers [Node 1]')

        plt.show()

if __name__ == '__main__':
    n = node0()
    n.main()