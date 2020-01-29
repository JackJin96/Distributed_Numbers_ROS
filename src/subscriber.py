#! /usr/bin/python

import rospy
from std_msgs.msg import Int64

def callback(msg):
    print "recieved " + str(msg.data)

def main():
    rospy.init_node('subscriber_node')
    sub = rospy.Subscriber('/numbers', Int64, callback)
    rospy.spin()

if __name__ == '__main__':
    main()