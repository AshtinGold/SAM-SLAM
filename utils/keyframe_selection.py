"""
Code for Keyframe Selection based on re-projection of points from 
the current frame to the keyframes.
"""

import torch
import numpy as np
import imageio


def get_pointcloud(depth, intrinsics, w2c, sampled_indices):
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices of sampled pixels
    # >> Sampled_indices is 2D coordinates of pixels
    xx = (sampled_indices[:, 1] - CX)/FX # >> pixel coordinates to normalized image coordinates using principal pt and focal len
    yy = (sampled_indices[:, 0] - CY)/FY
    depth_z = depth[0, sampled_indices[:, 0], sampled_indices[:, 1]]

    # Initialize point cloud
    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
    pts4 = torch.cat([pts_cam, torch.ones_like(pts_cam[:, :1])], dim=1)
    c2w = torch.inverse(w2c)
    pts = (c2w @ pts4.T).T[:, :3]

    # Remove points at camera origin
    A = torch.abs(torch.round(pts, decimals=4))
    B = torch.zeros((1, 3)).cuda().float()
    _, idx, counts = torch.cat([A, B], dim=0).unique(
        dim=0, return_inverse=True, return_counts=True)
    mask = torch.isin(idx, torch.where(counts.gt(1))[0])
    invalid_pt_idx = mask[:len(A)]
    valid_pt_idx = ~invalid_pt_idx
    pts = pts[valid_pt_idx]

    return pts


def keyframe_selection_overlap(gt_depth, w2c, intrinsics, keyframe_list, k, pixels=1600):
        """
        Select overlapping keyframes to the current camera observation.

        Args:
            gt_depth (tensor): ground truth depth image of the current frame.
            w2c (tensor): world to camera matrix (4 x 4).
            keyframe_list (list): a list containing info for each keyframe.
            k (int): number of overlapping keyframes to select.
            pixels (int, optional): number of pixels to sparsely sample 
                from the image of the current camera. Defaults to 1600.
        Returns:
            selected_keyframe_list (list): list of selected keyframe id.
        """
        # Radomly Sample Pixel Indices from valid depth pixels
        width, height = gt_depth.shape[2], gt_depth.shape[1]
        valid_depth_indices = torch.where(gt_depth[0] > 0)
        valid_depth_indices = torch.stack(valid_depth_indices, dim=1)
        indices = torch.randint(valid_depth_indices.shape[0], (pixels,))
        sampled_indices = valid_depth_indices[indices]

        # Back Project the selected pixels to 3D Pointcloud
        pts = get_pointcloud(gt_depth, intrinsics, w2c, sampled_indices)

        list_keyframe = []
        for keyframeid, keyframe in enumerate(keyframe_list):  ## keyframeid might not be the same as keyframe['id'] 
            # Get the estimated world2cam of the keyframe
            est_w2c = keyframe['est_w2c']
            # Transform the 3D pointcloud to the keyframe's camera space
            pts4 = torch.cat([pts, torch.ones_like(pts[:, :1])], dim=1)
            transformed_pts = (est_w2c @ pts4.T).T[:, :3]
            # Project the 3D pointcloud to the keyframe's image space
            points_2d = torch.matmul(intrinsics, transformed_pts.transpose(0, 1))
            points_2d = points_2d.transpose(0, 1)
            points_z = points_2d[:, 2:] + 1e-5  # ensure not zero-div
            points_2d = points_2d / points_z  # homogeneous coordinates
            projected_pts = points_2d[:, :2]
            # Filter out the points that are outside the image
            edge = 20
            mask = (projected_pts[:, 0] < width-edge)*(projected_pts[:, 0] > edge) * \
                (projected_pts[:, 1] < height-edge)*(projected_pts[:, 1] > edge)
            mask = mask & (points_z[:, 0] > 0)
            # Compute the percentage of points that are inside the image
            percent_inside = mask.sum()/projected_pts.shape[0]
            list_keyframe.append(
                {'id': keyframeid, 'percent_inside': percent_inside})

        # Sort the keyframes based on the percentage of points that are inside the image
        list_keyframe = sorted(
            list_keyframe, key=lambda i: i['percent_inside'], reverse=True)
        # Select the keyframes with percentage of points inside the image > 0
        selected_keyframe_ids = [keyframe_dict['id']
                                  for keyframe_dict in list_keyframe if keyframe_dict['percent_inside'] > 0.0]
        selected_keyframe_ids = list(np.random.permutation(
            np.array(selected_keyframe_ids))[:k])  ## select up to K

        return selected_keyframe_ids

#################
#
#
#
### DGS SLAM ###

def find_unique_class_rows(image):
    # Reshape the image array to have one row per pixel
    pixels = image.reshape(-1, 3) #  NM x 3
    uniq_ele, counts = torch.unique(pixels, dim=0, return_counts=True)
    count_sort_idx = torch.argsort(-counts) # sort by counts
    # print total classes
    print('total num classes: ', len(count_sort_idx))
    # top_uniq_vals = uniq_ele[count_sort_idx][:30]  # top 30 vals just to be safe
    top_uniq_vals = uniq_ele[count_sort_idx]

    return top_uniq_vals


def replace_non_matching_rows(array, row_value, replace_with):
    """
    Replace non-matching rows in the array with the specified row value.
    """
    # Create a boolean mask for rows that do not match the specified row value
    mask = torch.any(array != row_value, dim=1)
    mask = torch.tensor(mask[:,None]).to('cuda') # (1 x N) -> (N x 1)
    # print('\n mask'); print(mask)
    # Replace non-matching rows with the specified row value
    replace_with = torch.tensor(replace_with).to('cuda')

    modified_array = torch.where(mask[:, None], replace_with, array)
    raise

    return modified_array


def compute_IoU(im1, im2, rgb_val):
    """
    Compute IoU for a single rgb_val
    """
    # Reshape image for easier manipulation
    im1_vec = im1.reshape(-1, im1.shape[-1])
    im2_vec = im2.reshape(-1, im2.shape[-1])

    # isprint = 1
    # if isprint:
    #     print(im1_vec.shape)
    #     print(im2_vec.shape)

    # Mask out non-matching rows by indexing as either -1 or -2
    im1_vec = replace_non_matching_rows(im1_vec, rgb_val, np.array([-1, -1, -1]))
    im2_vec = replace_non_matching_rows(im2_vec, rgb_val, np.array([-2, -2, -2]))
    # compute IoU
    # intersection = (im1_vec == im2_vec).all(axis=1).sum()
    # Calculate element-wise equality between im1_vec and im2_vec

    equality_mask = torch.eq(im1_vec, im2_vec)
    # Check if all elements along axis 1 are True
    intersection = equality_mask.all(dim=1).sum()

    union = torch.logical_or(im1_vec == rgb_val, im2_vec == rgb_val).sum() / 3  # logical_or sums all 3 dims
    return intersection / union


def compute_mIoU(im1, im2):

    im1_rows = find_unique_class_rows(im1)  ## im1_cols: [30 x 3] # torchtensor
    im2_rows = find_unique_class_rows(im2)  ## im2_cols: [30 x 3] # torchtensor

    all_rows = torch.vstack((im1_rows, im2_rows))
    overall_unique_rows = torch.unique(all_rows, dim=0)

    sum_IoU = 0.0
    for row in overall_unique_rows:
        IoU = compute_IoU(im1, im2, row)
        sum_IoU += IoU

        raise
    mIoU = sum_IoU / overall_unique_rows.shape[0]
    return mIoU

# ---------------------------------
# ------------------
# NeurIPS 2021 paper "Few-Shot Segmentation via Cycle-Consistent Transformer
#-------------------------------

def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1).float()
    target = target.view(-1).float()
    output[target == ignore_index] = ignore_index
    intersection = output[output == target].float()
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K-1)
    area_output = torch.histc(output, bins=K, min=0, max=K-1)
    area_target = torch.histc(target, bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

def compute_miou_v2(im1, im2):
    K = int ( torch.max(torch.max(im1), torch.max(im2)) ) # highest lbl index
    i, u, _ = intersectionAndUnionGPU(im1, im2, K)
    classwise_IOU = i/u # tensor of size (num_classes)
    nan_indices = torch.isnan(classwise_IOU)
    classwise_IOU = classwise_IOU[~nan_indices]
    mIOU = classwise_IOU.sum()/classwise_IOU.shape[0] # mean IOU, taking (i/u).mean() is 
    return mIOU

#------------------------------
#-------------------------------


def keyframe_semantic_thresh_select(curr_semcol, keyframe_list, selected_keyframe_ids, thresh):
    """
    Log 2: Uses threshold for discarding.
    Log 1: Compares segmentation overlap by mIoU score between each keyframe and the current frame, discarding ones with high similarity.
    Args:
            curr_semcol (Keyframe): current frame semantic color
            keyframe_list (list): a list containing info for each keyframe.
            thresh: float = choose keyframes with score under threshold
    Returns:
            selected_keyframe_list (list): list of selected keyframe id.
    """
    # print('begin semantic keyframe search')
    new_keyframe_ids = []
    for _, keyframe in enumerate(keyframe_list):
        if keyframe['id'] in selected_keyframe_ids: # make sure we dont modify the original indexing
            keyframe_im = keyframe['sem_int_lbs']
            # keyframe_im = keyframe['sem_int_lbs'].permute(1,2,0)
            # curr_semcol = curr_semcol.permute(1,2,0)
            score = compute_miou_v2(curr_semcol, keyframe_im)
            # print(score) # is a tensor
            if score < thresh:  # what should thresh be?
                new_keyframe_ids.append(keyframe['id'])

    return new_keyframe_ids


# not implemented
def keyframe_semantic_bottom_k_discard(curr_frame, keyframe_list, selected_keyframe_ids):
    """
    Log 2: Discarding using top K mIOU scores
    Log 1: Computes segmentation overlap by mIoU.
    Similar to keyframe_semantic_thresh_select
    """
    # scoring_dict = {}  # will enter junk collection after calling this function
    # for _, keyframe in enumerate(keyframe_list):
    #     if keyframe['id'] in selected_keyframe_ids: # make sure we dont modify the original indexing
    #         score = compute_mIoU(curr_frame, keyframe_list)
    #         scoring_dict[keyframe['id'] : score]
    return