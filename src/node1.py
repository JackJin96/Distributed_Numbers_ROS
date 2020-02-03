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

class node1:
    def __init__(self):
        self.d = {}
        self.handled_ints = set({1, 2, 3, 4, 5})
        self.node_id = 1
        self.recieved_count = 0
        self.total_count = 0
        self.publish_rate = 500
        self.publish_queue_size = 500
        self.num_of_int = 5000

    def callback(self, feature):
        # print "\nNODE 1 CALLBACK:"
        data, from_id, to_id = feature.data, feature.header.frame_id, feature.to_id
        # rospy.loginfo("recieved: " + str(data))
        # rospy.loginfo("form_id: " + str(from_id))
        # rospy.loginfo("to_id: " + str(to_id))
        # self.total_count += 1
        if to_id == self.node_id:
            self.total_count += 1
            self.d[data] = self.d.get(data, 0) + 1
            # rospy.loginfo("current results: ")
            # rospy.loginfo(self.d)
        # rospy.loginfo("recieved " + str(self.recieved_count) + " messages")
        # rospy.loginfo("total published " + str(self.total_count) + " messages")

    def main(self):
        rospy.init_node('node1')
        
        sub = rospy.Subscriber('/numbers', Feature, self.callback)
        pub = rospy.Publisher('/numbers', Feature, queue_size = self.publish_queue_size)

        rate = rospy.Rate(self.publish_rate)
        feature = Feature()

        for i in range(self.num_of_int):
            # generate random integer form 1 to 10
            generated_int = random.randint(1, 10)

            # The integer generated is in the range for the node to process itself
            if generated_int in self.handled_ints:
                self.total_count += 1
                self.d[generated_int] = self.d.get(generated_int, 0) + 1
                # rospy.loginfo(self.d)
            # otherwise broadcast it in a message through rostopic
            else:
                feature.data = generated_int
                feature.header.frame_id = str(self.node_id)
                feature.to_id = 2
                # rospy.loginfo("publishing: " + str(feature.data))
                pub.publish(feature)
            rate.sleep()
        
        print "**********************************************"
        print "NODE 1 FINAL RESULT: "
        print "Total numbers processed: " + str(self.total_count)
        print "(number: frequency)"
        print self.d
        print "**********************************************"

        # Plot bar graph
        objects = [str(k) for k in self.d.keys()]
        y_pos = np.arange(len(objects))
        performance = [v for v in self.d.values()]

        plt.bar(y_pos, performance, align='center', alpha=0.5)
        plt.xticks(y_pos, objects)
        plt.ylabel('Frequency')
        plt.title('Frequency of random numbers [Node 1]')

        plt.show()

if __name__ == '__main__':
    n = node1()
    n.main()