#! /usr/bin/python

import rospy
import random
from std_msgs.msg import Int64

d = {}

def callback(msg):
    rospy.loginfo("recieved: " + str(msg.data))
    d[msg.data] = d.get(msg.data, 0) + 1
    rospy.loginfo("current results: ")
    rospy.loginfo(d)

def main():
    rospy.init_node('topic_publisher')
    
    sub = rospy.Subscriber('/numbers', Int64, callback)
    pub = rospy.Publisher('/numbers', Int64, queue_size=100)

    rate = rospy.Rate(1000)
    msg_int = Int64()

    for i in range(10000):
        msg_int = random.randint(1, 10)
        rospy.loginfo("published: " + str(msg_int))
        pub.publish(msg_int)
        rate.sleep()

if __name__ == '__main__':
    main()