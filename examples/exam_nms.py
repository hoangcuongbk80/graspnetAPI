__author__ = 'mhgou'
__version__ = '1.0'

# GraspNetAPI example for grasp nms.
# change the graspnet_root path

####################################################################
graspnet_root = '/media/cuong/HD-PZFU3/datasets/graspnet' # ROOT PATH FOR GRASPNET
####################################################################

sceneId = 88 #46 54 68 88
annId = 0 # 0 36 0 0

from graspnetAPI import GraspNet
import open3d as o3d
import cv2
import numpy as np

# initialize a GraspNet instance  
g = GraspNet(graspnet_root, camera='kinect', split='train')

# load grasps of scene 1 with annotation id = 3, camera = kinect and fric_coef_thresh = 0.2
_6d_grasp = g.loadGrasp(sceneId = sceneId, annId = annId, format = '6d', camera = 'kinect', fric_coef_thresh = 0.6)
#print('6d grasp:\n{}'.format(_6d_grasp))

# visualize the grasps using open3d
#geometries = []
#geometries.append(g.loadScenePointCloud(sceneId = sceneId, annId = annId, camera = 'kinect'))
#geometries += _6d_grasp.random_sample(numGrasp = 50).to_open3d_geometry_list()
#o3d.visualization.draw_geometries(geometries)

nms_grasp = _6d_grasp.nms(translation_thresh = 0.02, rotation_thresh = 60 / 180.0 * 3.1416)
#print('grasp after nms:\n{}'.format(nms_grasp))

# visualize the grasps using open3d
geometries = []
cloud = g.loadScenePointCloud(sceneId = sceneId, annId = annId, camera = 'kinect')
#cloud = cloud.voxel_down_sample(voxel_size=0.005)
#cloud = cloud.paint_uniform_color([1, 0.706, 0])
geometries.append(cloud)
#o3d.visualization.draw_geometries(geometries)


#o3d.visualization.draw_geometries(geometries)
#geometries += nms_grasp.to_open3d_geometry_list()
geometries += nms_grasp.random_sample(numGrasp = 60).to_open3d_geometry_list()
#o3d.visualization.draw_geometries(geometries)

vis1 = o3d.visualization.Visualizer()
vis1.create_window()
vis1.add_geometry(cloud)
opt1 = vis1.get_render_option()
opt1.point_size = 1
opt1.background_color = np.asarray([0, 0, 0])
vis1.run()
vis1.destroy_window()

vis2 = o3d.visualization.Visualizer()
vis2.create_window()
for geometry in geometries:
        vis2.add_geometry(geometry)
opt2 = vis2.get_render_option()
opt2.point_size = 1
opt2.background_color = np.asarray([0, 0, 0])
vis2.run()
vis2.destroy_window()


# show 6d poses
g.show6DPose(sceneIds = 88, show = True)