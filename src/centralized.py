#! /usr/bin/python

import rospy
import time
import random

rospy.init_node("centralized_node")

print "Running centralized_node"

numInt = 10000
d = {} # num: frequency

for i in range(numInt):
    rand_int = random.randint(1, 10)
    d[rand_int] = d.get(rand_int, 0) + 1

for k, v in d:
    print k, v