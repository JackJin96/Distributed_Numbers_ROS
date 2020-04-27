# Distributed Feature Matching in ROS

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

`node.py`:

- Extract features from images
  - Run k-means to partition features to each node
  - Keep features of images that are supposed to be handled by itself
  - Publish features of images that are supposed to be handled by other nodes
- Publish collected features to quickmatch_node for processing
  - Every second, the node would publish the features collected to the quickmatch_node and empty its collection

`quickmatch_node.py`:

- Upon receiving features in the callback function
  - Calculate distance matrix
  - Calculate feature density
  - Build tree
  - Sort edges
  - Break and merge tree
  - Graph results
  
  Click [here](https://docs.google.com/document/d/1KD0Jc04j5ioy37Hnn30I92voBEUOVhij1nFr4NVViA4/edit?usp=sharing) for complete documentation.
