import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
from datasets import find_dataset_def
from models import *
from utils import *
import sys
from datasets.data_io import read_pfm, save_pfm
import cv2
from plyfile import PlyData, PlyElement
from PIL import Image

import shutil
import subprocess
import trimesh

# model
loadckpt = "checkpoints/d192/model_000014.ckpt"

model = MVSNet(refine=False)
model = nn.DataParallel(model)
model.cuda()

# load checkpoint file specified by args.loadckpt
print("loading model {}".format(loadckpt))
state_dict = torch.load(loadckpt)
model.load_state_dict(state_dict['model'])
model.eval()

# Warm up process
# Dummy warm-up image
dummy_input_1 = torch.randn(1, 5, 3, 896, 1600)
dummy_input_2 = torch.randn(1, 5, 4, 4)
dummy_input_3 = torch.randn(1, 192)

dummy_input_1 = tocuda(dummy_input_1)
dummy_input_2 = tocuda(dummy_input_2)
dummy_input_3 = tocuda(dummy_input_3)

# Warm-up phase
with torch.no_grad():
    _ = model(dummy_input_1, dummy_input_2, dummy_input_3)


# read intrinsics and extrinsics
def read_camera_parameters(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    # TODO: assume the feature is 1/4 of the original image size
    intrinsics[:2, :] /= 4
    return intrinsics, extrinsics

# read an image
def read_img(filename):
    img = Image.open(filename)
    # scale 0~255 to 0~1
    np_img = np.array(img, dtype=np.float32) / 255.
    return np_img

# read a binary mask
def read_mask(filename):
    return read_img(filename) > 0.5

# save a binary mask
def save_mask(filename, mask):
    # assert mask.dtype == np.bool
    assert mask.dtype == bool
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(filename)

# Preprocess dataset using colmap
def preproc_colmap(dataset="handshake_phase2_100frames", input_folder="mvs_testing", frame="901"):

    input_path = os.path.join(input_folder, dataset, frame)

    image_dir = os.path.join(input_path, "images")
    os.makedirs(image_dir, exist_ok=True)

    sparse_dir = os.path.join(input_path, "sparse")
    os.makedirs(sparse_dir, exist_ok=True)

    num_images = 5
    for i in range(num_images):
        from_dir = os.path.join(input_path, f"{i}.jpg")
        shutil.copy(from_dir, image_dir)

    commands = (
            f"colmap feature_extractor --database_path {input_path}/database.db --image_path {image_dir};" + \
            f"colmap exhaustive_matcher --database_path {input_path}/database.db;" + \
            f"colmap mapper --database_path {input_path}/database.db --image_path {image_dir} --output_path {sparse_dir};" + \
            f"colmap model_converter --input_path {sparse_dir}/0 --output_path {sparse_dir} --output_type TXT;" + \
            f"python3 colmap2mvsnet.py --dense_folder {input_path} --max_d 192;")

    commands_backup = (
            f"colmap mapper --database_path {input_path}/database.db --image_path {image_dir} --output_path {sparse_dir};" + \
            f"colmap model_converter --input_path {sparse_dir}/1 --output_path {sparse_dir} --output_type TXT;" + \
            f"python3 colmap2mvsnet.py --dense_folder {input_path} --max_d 192;")

    try:
        subprocess.check_output(commands, shell=True)
    except subprocess.CalledProcessError:
        #print(e.output)
        with open(f"log.txt", "a") as f:
            f.write(image_dir + "\n")
        subprocess.run(commands_backup, shell=True)

    print("Done preprocessing")


# run MVS model to save depth maps and confidence maps
def save_depth(dataset="handshake_phase2_100frames", input_folder="mvs_testing", output_folder="mvsnet_outputs", frame="901", cams=[1, 1, 1, 1, 1]):

    numdepth = 192
    interval_scale = 1.06
    batch_size = 1

    input_path = os.path.join(input_folder, dataset)
    testlist = os.path.join(input_path, "test_list.txt")

    with open(testlist, "w") as f:
        f.write(frame)
        f.write("\n")

    num_cams = cams.count(1)

    # dataset, dataloader
    MVSDataset = find_dataset_def("dtu_mvs")
    test_dataset = MVSDataset(input_path, testlist, "test", num_cams, cams, numdepth, interval_scale)
    TestImgLoader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=4, drop_last=False)
    pair = TestImgLoader.dataset.metas

    outdir = os.path.join(output_folder, dataset)

    with torch.no_grad():
        for batch_idx, sample in enumerate(TestImgLoader):
            sample_cuda = tocuda(sample)
            outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
            outputs = tensor2numpy(outputs)
            del sample_cuda
            print('Iter {}/{}'.format(batch_idx, len(TestImgLoader)))
            filenames = sample["filename"]

            # save depth maps and confidence maps
            for filename, depth_est, photometric_confidence in zip(filenames, outputs["depth"],
                                                                   outputs["photometric_confidence"]):
                depth_filename = os.path.join(outdir, filename.format('depth_est', '.pfm'))
                confidence_filename = os.path.join(outdir, filename.format('confidence', '.pfm'))
                os.makedirs(depth_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(confidence_filename.rsplit('/', 1)[0], exist_ok=True)
                # save depth maps
                save_pfm(depth_filename, depth_est)
                # save confidence maps
                save_pfm(confidence_filename, photometric_confidence)

    print("done save depth")
    return pair


# project the reference point cloud into the source view, then project back
def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    ## step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    # reference 3D space
    xyz_ref = np.matmul(np.linalg.inv(intrinsics_ref),
                        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]))
    # source 3D space
    xyz_src = np.matmul(np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)),
                        np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    # source view x, y
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)
    # mask = sampled_depth_src > 0

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = np.matmul(np.linalg.inv(intrinsics_src),
                        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
    # reference 3D space
    xyz_reprojected = np.matmul(np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
                                np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(depth_ref,
                                                                                                 intrinsics_ref,
                                                                                                 extrinsics_ref,
                                                                                                 depth_src,
                                                                                                 intrinsics_src,
                                                                                                 extrinsics_src)
    # check |p_reproj-p_1| < 1
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref

    mask = np.logical_and(dist < 1, relative_depth_diff < 0.01)
    depth_reprojected[~mask] = 0

    return mask, depth_reprojected, x2d_src, y2d_src


def filter_depth(dataset, input_folder, output_folder, frame, pair_data):
    display = False

    scan_folder = os.path.join(input_folder, dataset, frame)

    out_folder = os.path.join(output_folder, dataset, frame)
    plyfilename = os.path.join(output_folder, dataset, f"mvsnet_{frame}_l3.ply")

    # for the final point cloud
    vertexs = []
    vertex_colors = []

    # for each reference view and the corresponding source views
    for frame, ref_view, src_views in pair_data:
        # load the camera parameters
        ref_intrinsics, ref_extrinsics = read_camera_parameters(
            os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(ref_view)))
        # load the reference image
        ref_img = read_img(os.path.join(scan_folder, 'images/{:0>8}.jpg'.format(ref_view)))
        # load the estimated depth of the reference view
        ref_depth_est = read_pfm(os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(ref_view)))[0]
        # load the photometric mask of the reference view
        confidence = read_pfm(os.path.join(out_folder, 'confidence/{:0>8}.pfm'.format(ref_view)))[0]
        photo_mask = confidence > 0.8

        all_srcview_depth_ests = []
        all_srcview_x = []
        all_srcview_y = []
        all_srcview_geomask = []

        # compute the geometric mask
        geo_mask_sum = 0
        for src_view in src_views:
            # camera parameters of the source view
            src_intrinsics, src_extrinsics = read_camera_parameters(
                os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(src_view)))
            # the estimated depth of the source view
            src_depth_est = read_pfm(os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(src_view)))[0]

            geo_mask, depth_reprojected, x2d_src, y2d_src = check_geometric_consistency(ref_depth_est, ref_intrinsics,
                                                                                        ref_extrinsics,
                                                                                        src_depth_est,
                                                                                        src_intrinsics, src_extrinsics)
            geo_mask_sum += geo_mask.astype(np.int32)
            all_srcview_depth_ests.append(depth_reprojected)
            all_srcview_x.append(x2d_src)
            all_srcview_y.append(y2d_src)
            all_srcview_geomask.append(geo_mask)

        depth_est_averaged = (sum(all_srcview_depth_ests) + ref_depth_est) / (geo_mask_sum + 1)
        # at least 3 source views matched
        geo_mask = geo_mask_sum >= 3
        final_mask = np.logical_and(photo_mask, geo_mask)

        os.makedirs(os.path.join(out_folder, "mask"), exist_ok=True)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_photo.png".format(ref_view)), photo_mask)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_geo.png".format(ref_view)), geo_mask)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_final.png".format(ref_view)), final_mask)

        print("processing {}, ref-view{:0>2}, photo/geo/final-mask:{}/{}/{}".format(scan_folder, ref_view,
                                                                                    photo_mask.mean(),
                                                                                    geo_mask.mean(), final_mask.mean()))

        if display:
            import cv2
            cv2.imshow('ref_img', ref_img[:, :, ::-1])
            cv2.imshow('ref_depth', ref_depth_est / 800)
            cv2.imshow('ref_depth * photo_mask', ref_depth_est * photo_mask.astype(np.float32) / 800)
            cv2.imshow('ref_depth * geo_mask', ref_depth_est * geo_mask.astype(np.float32) / 800)
            cv2.imshow('ref_depth * mask', ref_depth_est * final_mask.astype(np.float32) / 800)
            cv2.waitKey(0)

        height, width = depth_est_averaged.shape[:2]
        x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
        # valid_points = np.logical_and(final_mask, ~used_mask[ref_view])
        valid_points = final_mask
        print("valid_points", valid_points.mean())
        x, y, depth = x[valid_points], y[valid_points], depth_est_averaged[valid_points]

        #color = ref_img[1:-16:4, 1::4, :][valid_points]  # hardcoded for DTU dataset
        color = ref_img[1:-4:4, 1::4, :][valid_points]  # hardcoded for our dataset

        #color = ref_img[1:-2:4, 1::4, :][valid_points]  # hardcoded for our new 896 dataset

        xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics),
                            np.vstack((x, y, np.ones_like(x))) * depth)
        xyz_world = np.matmul(np.linalg.inv(ref_extrinsics),
                              np.vstack((xyz_ref, np.ones_like(x))))[:3]
        vertexs.append(xyz_world.transpose((1, 0)))
        vertex_colors.append((color * 255).astype(np.uint8))

    vertexs = np.concatenate(vertexs, axis=0)
    vertex_colors = np.concatenate(vertex_colors, axis=0)
    vertexs = np.array([tuple(v) for v in vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(plyfilename)
    print("saving the final model to", plyfilename)


# Evaluation metric using color
import math
def read_cam_file(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]

    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))

    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    #intrinsics[:2, :] /= 4

    interval_scale = 1.06

    # depth_min & depth_interval: line 11
    depth_min = float(lines[11].split()[0])
    depth_interval = float(lines[11].split()[1]) * interval_scale
    return intrinsics, extrinsics, depth_min, depth_interval

import matplotlib.pyplot as plt
def project_points_to_image(points_3d, point_color, intrinsics, extrinsics, image_size):

    image_height, image_width = image_size
    point_color = point_color[:, :3]

    # Convert points to homogeneous coordinates
    num_points = points_3d.shape[0]
    points_3d_hom = np.hstack((points_3d, np.ones((num_points, 1))))

    # Transform points to camera coordinates
    points_cam = (extrinsics @ points_3d_hom.T).T

    # Project points onto the image plane using the intrinsic matrix
    points_2d_hom = (intrinsics @ points_cam[:, :3].T).T

    # Convert from homogeneous to 2D coordinates
    #points_2d = points_2d_hom[:, :2] / points_2d_hom[:, 2, np.newaxis]
    u = points_2d_hom[:, 0] / points_2d_hom[:, 2]
    v = points_2d_hom[:, 1] / points_2d_hom[:, 2]

    Z_c = points_cam[:, 2]

    # Round and convert to integers
    u = np.round(u).astype(int)
    v = np.round(v).astype(int)

    # Filter Valid Points
    valid = (Z_c > 0) & (u >= 0) & (u < image_width) & (v >= 0) & (v < image_height)
    u = u[valid]
    v = v[valid]
    Z_c = Z_c[valid]
    point_color = point_color[valid]

    # Handle Depth Buffering with Vectorized Operations
    depth_buffer = np.full((image_height, image_width), np.inf)
    image = np.zeros((image_height, image_width, 3), dtype=np.float32)

    # Get current depth values at pixel locations
    current_depths = depth_buffer[v, u]

    # Create a mask where the new depth is less than the current depth
    update_mask = Z_c < current_depths

    # Update depth buffer and image where the new depth is closer
    depth_buffer[v[update_mask], u[update_mask]] = Z_c[update_mask]
    image[v[update_mask], u[update_mask]] = point_color[update_mask]

    # cv2.imshow('test', image)
    # cv2.waitKey(0)

    # plt.imshow(image/255)
    # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    # plt.show()

    #print("Done evlutea point")
    return image


def colour_distance(sparse_image, original_image):

    valid_mask = np.any(sparse_image != [0, 0, 0], axis=-1)

    sparse_colors = sparse_image[valid_mask]
    original_colors = original_image[valid_mask]

    r1 = sparse_colors[:, 0]
    g1 = sparse_colors[:, 1]
    b1 = sparse_colors[:, 2]

    r2 = original_colors[:, 0]
    g2 = original_colors[:, 1]
    b2 = original_colors[:, 2]

    r_mean = (r1 + r2) / 2
    r = r1 - r2
    g = g1 - g2
    b = b1 - b2

    dis = np.sqrt(np.multiply(2 + r_mean / 256, np.multiply(r, r))
                  + 4 * np.multiply(g, g) +
                  np.multiply(2 + (255 - r_mean) / 256, np.multiply(b, b)))

    return dis

def eval_metric_color(dataset, input_folder, output_folder, cams, frame, gt_frame):

    num_points = []

    for view_id, view in enumerate(cams):

        if view == 0:
            num_points.append(0)
            continue

        cam_file = os.path.join(input_folder, dataset, frame, "cams", f"0000000{view_id}_cam.txt")
        intrinsics, extrinsics, depth_min, depth_interval = read_cam_file(cam_file)

        pc_file = os.path.join(output_folder, dataset, f"mvsnet_{frame}_l3.ply")
        pc = trimesh.load(pc_file)
        point_clouds = pc.vertices.view(np.ndarray)
        vertex_colors = pc.visual.vertex_colors

        image_gt = cv2.imread(os.path.join(input_folder, dataset, gt_frame, "images", f"0000000{view_id}.jpg"))
        gt_image_rgb = cv2.cvtColor(image_gt, cv2.COLOR_BGR2RGB)
        gt_image_rgb = gt_image_rgb.astype(np.float32)

        image_height, image_width, channel = gt_image_rgb.shape
        project_image = project_points_to_image(point_clouds, vertex_colors, intrinsics, extrinsics, (image_height, image_width))

        color_diff = colour_distance(project_image, gt_image_rgb)
        # Define the threshold
        threshold = 50

        matching_points = np.sum(color_diff <= threshold)
        num_points.append(matching_points)

    print("done evaluation")
    return num_points


if __name__ == '__main__':

    cams = [1, 1, 1, 1, 1]

    dataset = "handshake_phase2_100frames"
    input_folder = "mvs_testing"
    output_folder = "mvsnet_outputs"

    frame = "901"
    gt_frame = "901"

    start_time = time.time()

    cur_time = time.time()
    preproc_colmap(dataset, input_folder, frame)
    preproc_cost_time = time.time() - cur_time

    cur_time = time.time()
    pair_data = save_depth(dataset, input_folder, output_folder, frame, cams)
    save_depth_cost_time = time.time() - cur_time

    cur_time = time.time()
    filter_depth(dataset, input_folder, output_folder, frame, pair_data)
    filter_depth_cost_time = time.time() - cur_time

    cur_time = time.time()
    num_points_from_views = eval_metric_color(dataset, input_folder, output_folder, cams, frame, gt_frame)
    eval_cost_time = time.time() - cur_time

    total_cost_time = time.time() - start_time
    print("done")
