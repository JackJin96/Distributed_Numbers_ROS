import yaml
import numpy as np

new_label_matrix = np.random.rand(3, 128)

with open('/home/tron_ubuntu2/catkin_ws/src/dist_num/config/params.yaml','r') as yamlfile:
    cur_yaml = yaml.safe_load(yamlfile) # Note the safe_load
    cur_yaml['label_matrix'] = new_label_matrix.tolist()

if cur_yaml:
    with open('/home/tron_ubuntu2/catkin_ws/src/dist_num/config/params.yaml','w') as yamlfile:
        yaml.safe_dump(cur_yaml, yamlfile) # Also note the safe_dump