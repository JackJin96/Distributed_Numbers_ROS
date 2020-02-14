# Distributed Random Number Frequency Counter in ROS

## First Version: Distributed Random Number Frequency Counter

- Random Numbers from 1 to 10
- Two nodes:
  - Node1 handles 1 to 5
  - Node2 handles 6 to 10

## Second Version: Distributed Nearest Neighbor Classifier

- Random Vectors [x1, x2] with -5 <= x1 <= 2, and -5 <= x2 <= 25
- Three Nodes
  - Node0: [0, 0]
  - Node1: [10, 10]
  - Node2: [20, 20]

## Third Version: Distrubuted Features

Currently, the node:

- Extract features from images
  - Keep features of images that are supposed to be handled by itself
  - Publish features of images that are supposed to be handled by other nodes

## How to create custom messages in ROS

http://wiki.ros.org/ROS/Tutorials/CreatingMsgAndSrv#Creating_a_msg

http://wiki.ros.org/ROS/Tutorials/CustomMessagePublisherSubscriber%28python%29
