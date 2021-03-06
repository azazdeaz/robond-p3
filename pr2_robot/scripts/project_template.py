#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml

SCENE_NUM = 2


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

# Exercise-2 TODOs:

    # TODO: Convert ROS msg to PCL data
    cloud = ros_to_pcl(pcl_msg)

    # TODO: Statistical Outlier Filtering
    outlier_filter = cloud.make_statistical_outlier_filter()
    outlier_filter.set_mean_k(50)
    outlier_filter.set_std_dev_mul_thresh(0.1)
    cloud = outlier_filter.filter()

    # TODO: Voxel Grid Downsampling
    vox = cloud.make_voxel_grid_filter()
    LEAF_SIZE = 0.01
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    cloud = vox.filter()

    # TODO: PassThrough Filter
    passthrough_y = cloud.make_passthrough_filter()
    passthrough_y.set_filter_field_name('y')
    passthrough_y.set_filter_limits(-0.4, 0.4)
    cloud = passthrough_y.filter()

    passthrough_z = cloud.make_passthrough_filter()
    passthrough_z.set_filter_field_name('z')
    passthrough_z.set_filter_limits(0.6, 0.9)
    cloud = passthrough_z.filter()


    # TODO: RANSAC Plane Segmentation
    seg = cloud.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(0.01)
    inliers, coefficients = seg.segment()
    cloud = cloud.extract(inliers, negative=True)

    # TODO: Extract inliers and outliers

    # TODO: Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(cloud)
    tree = white_cloud.make_kdtree()
    ec = white_cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(0.01)
    ec.set_MinClusterSize(100)
    ec.set_MaxClusterSize(250000)
    ec.set_SearchMethod(tree)
    cluster_indices = ec.Extract()

    # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately
    cluster_color = get_color_list(len(cluster_indices))
    color_cluster_point_list = []
    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                            white_cloud[indice][1],
                                            white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])

    #Create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

    # TODO: Convert PCL data to ROS messages
    pcl_msg = pcl_to_ros(cluster_cloud)

    # TODO: Publish ROS messages
    pcl_objects_pub.publish(pcl_msg)

# Exercise-3 TODOs:

    # Publish the list of detected objects
    detected_objects_labels = []
    detected_objects = []

    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster from the extracted outliers (cloud_objects)
        # print('cloud l', len(pts_list))
        pcl_cluster = cloud.extract(pts_list)
        pcl_object_pubs[index].publish(pcl_to_ros(pcl_cluster))
        # TODO: convert the cluster from pcl to ROS using helper function
        pcl_msg = pcl_to_ros(pcl_cluster)
        # Extract histogram features
        # TODO: complete this step just as is covered in capture_features.py
        chists = compute_color_histograms(pcl_msg, using_hsv=True)
        normals = get_normals(pcl_msg)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))

        # Make the prediction, retrieve the label for the result
        # and add it to detected_objects_labels list
        # print('feature.reshape(1,-1)', feature.reshape(1,-1))
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = pcl_msg
        detected_objects.append(do)

    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))

    # Publish the list of detected objects
    # This is the output you'll need to complete the upcoming project!
    detected_objects_pub.publish(detected_objects)

    object_list_param = rospy.get_param('/object_list')
    dropbox_param = rospy.get_param('/dropbox')
    dropbox = dict(zip([box['group'] for box in dropbox_param], dropbox_param))

    labels = []
    centroids = []
    yaml_list = []

    for do in detected_objects:
        labels.append(do.label)
        points_arr = ros_to_pcl(do.cloud).to_array()
        centroids.append(np.mean(points_arr, axis=0)[:3].tolist())

    for i, param in enumerate(object_list_param):
        test_scene_num = Int32()
        test_scene_num.data = SCENE_NUM
        object_name = String()
        object_name.data = param['name']
        arm_name = String()
        arm_name.data = param['group']
        pick_pose = Pose()
        place_pose = Pose()

        if param['name'] in detected_objects_labels:
            detected_index = detected_objects_labels.index(param['name'])
            print('detected', i, detected_index, param['name'])
            centroid = centroids[detected_index]
            pick_pose.position.x = centroid[0]
            pick_pose.position.y = centroid[1]
            pick_pose.position.z = centroid[2]
            place_pose.position.x = dropbox[param['group']]['position'][0]
            place_pose.position.y = dropbox[param['group']]['position'][1]
            place_pose.position.z = dropbox[param['group']]['position'][2]

            yaml = make_yaml_dict(test_scene_num=test_scene_num,
                                  object_name=object_name,
                                  arm_name=arm_name,
                                  pick_pose=pick_pose,
                                  place_pose=place_pose)
            yaml_list.append(yaml)
        else:
            print('Cant find "{}"'.format(param["name"]))
    send_to_yaml('output_{}.yaml'.format(SCENE_NUM), yaml_list)




    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    # try:
    #     pr2_mover(detected_objects_list)
    # except rospy.ROSInterruptException:
    #     pass

# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    # TODO: Initialize variables

    # TODO: Get/Read parameters

    # TODO: Parse parameters into individual variables

    # TODO: Rotate PR2 in place to capture side tables for the collision map

    # TODO: Loop through the pick list

        # TODO: Get the PointCloud for a given object and obtain it's centroid

        # TODO: Create 'place_pose' for the object

        # TODO: Assign the arm to be used for pick_place

        # TODO: Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format

        # Wait for 'pick_place_routine' service to come up
        rospy.wait_for_service('pick_place_routine')

        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            # TODO: Insert your message variables to be sent as a service request
            resp = pick_place_routine(TEST_SCENE_NUM, OBJECT_NAME, WHICH_ARM, PICK_POSE, PLACE_POSE)

            print ("Response: ",resp.success)

        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    # TODO: Output your request parameters into output yaml file



if __name__ == '__main__':

    # TODO: ROS node initialization
    rospy.init_node('robot')

    # TODO: Create Subscribers
    pcl_sub = rospy.Subscriber('pr2/world/points', pc2.PointCloud2, pcl_callback, queue_size=1)

    # TODO: Create Publishers
    pcl_objects_pub = rospy.Publisher('pcl_objects_color_groups', pc2.PointCloud2, queue_size=1)
    detected_objects_pub = rospy.Publisher('detected_objects', DetectedObjectsArray, queue_size=1)
    object_markers_pub = rospy.Publisher('object_markers', Marker, queue_size=1)
    pcl_object_pubs = [
        rospy.Publisher('pcl_object_1', pc2.PointCloud2, queue_size=1),
        rospy.Publisher('pcl_object_2', pc2.PointCloud2, queue_size=1),
        rospy.Publisher('pcl_object_3', pc2.PointCloud2, queue_size=1),
        rospy.Publisher('pcl_object_4', pc2.PointCloud2, queue_size=1),
        rospy.Publisher('pcl_object_5', pc2.PointCloud2, queue_size=1),
        rospy.Publisher('pcl_object_6', pc2.PointCloud2, queue_size=1),
        rospy.Publisher('pcl_object_7', pc2.PointCloud2, queue_size=1),
        rospy.Publisher('pcl_object_8', pc2.PointCloud2, queue_size=1),
    ]

    # TODO: Load Model From disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
