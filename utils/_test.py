from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
import torch

from time import time

test_image_path = "/mnt/c2d9b23a-b03e-4fdb-82ad-59f039ec9e3e/intern/King_Hang/LangSplat/rm0_300/images/frame000000.jpg"  # test image
image = cv2.imread(test_image_path)

print("image.shape: ")
print(image.shape)

# resize if too large
orig_w, orig_h = image.shape[1], image.shape[0]
global_down = 1
if orig_h > 1080:
    global_down = orig_h / 1080

scale = float(global_down)
resolution = (int( orig_w  / scale), int(orig_h / scale))
image = cv2.resize(image, resolution)


sam_ckpt_path = "/mnt/c2d9b23a-b03e-4fdb-82ad-59f039ec9e3e/intern/King_Hang/LangSplat/ckpts/sam_vit_h_4b8939.pth"
sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt_path).to('cuda')
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.7,
    box_nms_thresh=0.7,
    stability_score_thresh=0.85,
    crop_n_layers=1,
    crop_n_points_downscale_factor=1,
    min_mask_region_area=100,
)

print("image.shape AFTER resizing: ")
print(image.shape)

starttime = time() 

masks_default, masks_s, masks_m, masks_l = mask_generator.generate(image)
print('#######')
print('CODE IS RUN SUCCESSFULLY')

print('Timing required :')
endtime = time()
print(endtime  - starttime)