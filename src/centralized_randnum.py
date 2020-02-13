#! /usr/bin/python

import rospy
import time
import random
# import cv2

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

rospy.init_node("centralized_node")

print "Running centralized_node"
print rospy.get_param(rospy.get_name())

numInt = 10000
d = {} # num: frequency

for i in range(numInt):
    rand_int = random.randint(1, 10)
    d[rand_int] = d.get(rand_int, 0) + 1

print d

# Plot bar graph
objects = [str(k) for k in d.keys()]
y_pos = np.arange(len(objects))
performance = [v for v in d.values()]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Frequency')
plt.title('Frequency of random numbers [Centralized]')

plt.show()