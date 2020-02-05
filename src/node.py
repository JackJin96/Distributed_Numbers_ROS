#! /usr/bin/python

'''
Node 1 handles numbers from 1 to 5
'''

import time
import rospy
import random
import sys

from std_msgs.msg import Int64
from dist_num.msg import Feature # pylint: disable = import-error

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

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
        min_dist = float('inf')
        res_index = -sys.maxint - 1
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
            self.collection.append(list(data))

    def main(self):
        sub = rospy.Subscriber('/numbers', Feature, self.callback)
        pub = rospy.Publisher('/numbers', Feature, queue_size = self.publish_queue_size)

        rate = rospy.Rate(self.publish_rate)
        feature = Feature()

        # pausing for 1 second before start can ensure the first messages are not lost
        # otherwise the first messages are lost
        # number of lost messages = number of nodes initialized
        time.sleep(1)

        for i in range(self.amount_generated):
            # generate random vector with set higher bound and lower bound
            generated_vector = [random.uniform(self.lower_bound, self.higher_bound), random.uniform(self.lower_bound, self.higher_bound)]
            belongs_to_index = self.closest_vector(generated_vector, self.matrix)
            print 'generated vector:'
            print generated_vector
            print 'belongs to index:'
            print belongs_to_index

            # The integer generated is in the range for the node to process itself
            if belongs_to_index == self.node_id:
                self.total_count += 1
                self.collection.append(generated_vector)
            # otherwise broadcast it in a message through rostopic
            else:
                feature.data = generated_vector
                feature.header.frame_id = str(self.node_id)
                feature.to_id = belongs_to_index
                pub.publish(feature)
            rate.sleep()
        
        print "**********************************************"
        print rospy.get_name() + " FINAL RESULT: "
        print "Total numbers processed: " + str(self.total_count)
        print "(number: frequency)"
        print self.collection
        print "**********************************************"

        # Plot scattered points graph for distribution
        na = np.array(self.collection)
        fig, ax = plt.subplots()
        ax.scatter(na[:, 0], na[:, 1], c='blue')
        for i in range(len(self.matrix)):
            ax.scatter(self.matrix[i][0], self.matrix[i][1], c="red")
        line1 = mlines.Line2D([-5, 15], [15, -5])
        line2 = mlines.Line2D([30, 0], [0, 30])
        ax.add_line(line1)
        ax.add_line(line2)
        plt.xlim(self.lower_bound, self.higher_bound)
        plt.ylim(self.lower_bound, self.higher_bound)
        plt.title('node ' + str(self.node_id) + ', count: ' + str(self.total_count))
        plt.show()

if __name__ == '__main__':
    n = node0()
    n.main()