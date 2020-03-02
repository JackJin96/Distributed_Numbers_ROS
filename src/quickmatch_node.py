#!/usr/bin/env python

import rospy
from std_msgs.msg import String


def callback(msg):
    rospy.loginfo(' I heard %s', msg.data)

def main():
    rospy.init_node('quickmatch_node')

    sub = rospy.Subscriber('/publishStatus', String, callback)
    #Publish on the 'chatter' topic
    pub = rospy.Publisher('/publishStatus', String, queue_size=10)

    #Prepare message object
    msg = String()

    #Set rate to use (in Hz)
    rate = rospy.Rate(1)

    i = 0
    #Use sleep to allow subscriber to set up in a period of time
    rate.sleep()
    while not rospy.is_shutdown():
        #Write to console
        rospy.loginfo(msg.data)
        #Publish
        pub.publish(msg.data)
        #Wait until it is done
        rate.sleep()


if __name__ == '__main__':
    main()