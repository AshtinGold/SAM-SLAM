import sys, os
import torch
import numpy as np
import torch.nn as nn
import open_clip
import argparse
from importlib.machinery import SourceFileLoader
from plyfile import PlyData, PlyElement

# add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '../autoencoder'))
from model import Autoencoder
sys.path.append(os.path.join(os.path.dirname(__file__), '../scripts'))
from export_ply import save_ply


def main():
    """
    # How to identify closest CLIP embed
    1. text to CLIP
    2. CLIP -> autoencoder
    3. search against all entries in param up to a norm distance threshold. Since for an object, the 3-dim LE is the same, we do not need to worry if we accidentally crop out part of the object.
    4. Search across all 3 levels of scale
    5. Save the search result to a new param file.
    """

    query_str  : str = "A chair" #>>
    
    model, _, _ = open_clip.create_model_and_transforms(
    model_name = 'ViT-B-16',  # e.g., ViT-B-16
    pretrained= 'laion2b_s34b_b88k',
    precision="fp16",
    )
    model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    
    text = tokenizer(["a diagram", "a dog", "a cat"])
    #>>
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    print('Raw CLIP feature: ')
    print(text_features)
    
    #
    checkpoint_path = '/mnt/c2d9b23a-b03e-4fdb-82ad-59f039ec9e3e/intern/King_Hang/OFFLINE-samslam_2/autoencoder/ckpt/rm0/best_ckpt.pth'
    model = setup_autoencoder(checkpoint_path)

    dim3_out = model.encode(text).to("cpu").numpy()  
    print('Encoded CLIP feature: ')
    print(dim3_out)
    return

def setup_autoencoder(checkpoint_path):
    encoder_hidden_dims = [256, 128, 64, 32, 3]
    decoder_hidden_dims=[16, 32, 64, 128, 256, 256, 512]
    checkpoint_path = '/mnt/c2d9b23a-b03e-4fdb-82ad-59f039ec9e3e/intern/King_Hang/OFFLINE-samslam_2/autoencoder/ckpt/rm0/best_ckpt.pth'
    checkpoint = torch.load(checkpoint_path)
    model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to("cuda:0")
    model.load_state_dict(checkpoint)
    model.eval()

    return model

def forward_process(text:str):
    import torch
    from PIL import Image
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    text = tokenizer([text])

    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    print('CLIP features original DIM')
    # print(text_features)
    print(text_features.shape) #>> torch.Size([1, 512])
    print(type(text_features))

    checkpoint_path = '/mnt/c2d9b23a-b03e-4fdb-82ad-59f039ec9e3e/intern/King_Hang/OFFLINE-samslam_2/autoencoder/ckpt/rm0/best_ckpt.pth'
    model = setup_autoencoder(checkpoint_path)

    model = model.to("cuda")
    text_features = text_features.to("cuda")

    dim3_out = model.encode(text_features).to('cpu').detach().numpy()  
    print('Encoded CLIP feature: ')
    # # print(dim3_out)
    # print(dim3_out.shape)
    print(dim3_out)

    return dim3_out

def text2embed3d(text):
    return forward_process(text)

def reverse_process():
    """ From dim3 to 512D"""
    raise NotImplementedError

def query_ptcd(query_tensor, le_s):
    _, indices = get_cosine_similarity(query_tensor, le_s)
    return indices

def get_cosine_similarity(query_tensor, le_s, cos_threshold = 0.7):

    # Convert numpy arrays to tensors if needed
    if isinstance(query_tensor, np.ndarray):
        query_tensor = torch.from_numpy(query_tensor)
    if isinstance(le_s, np.ndarray):
        le_s = torch.from_numpy(le_s)
    
    query_tensor = torch.tensor(query_tensor, dtype=torch.float32)
    le_s = torch.tensor(le_s, dtype=torch.float32)
    
    # check dimensions
    if query_tensor.dim() != 2 or le_s.dim() != 2:
        raise ValueError("Both query_tensor and le_s should be 2D tensors.")
    if query_tensor.size(1) != le_s.size(1):
        raise ValueError("The second dimension of query_tensor and le_s must be the same.")

    norm_query_tensor = query_tensor / query_tensor.norm(dim=1, keepdim=True)
    norm_le_s = le_s / le_s.norm(dim=1, keepdim=True)
    
    cosine_similarities = torch.mm(norm_query_tensor, norm_le_s.t())
    indices = (cosine_similarities.squeeze(0) > cos_threshold).nonzero(as_tuple=True)[0]
    
    return cosine_similarities, indices


def load_pointcloud(params_dir):
    return np.load(params_dir)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config file.")
    return parser.parse_args()

if __name__ == '__main__':
    """
    USAGE:
    python utils/semantic_query.py configs/replica/splatam.py

    # How to identify closest CLIP embed
    1. text to CLIP
    2. CLIP -> autoencoder
    3. search against all entries in param up to a norm distance threshold. Since for an object, the 3-dim LE is the same, we do not need to worry if we accidentally crop out part of the object.
    4. Search across all 3 levels of scale
    5. Save the search result to a new param file.
    """
    args = parse_args()

    # Load SplaTAM config
    experiment = SourceFileLoader(os.path.basename(args.config), args.config).load_module()
    config = experiment.config
    work_path = config['workdir']
    run_name = config['run_name']

    text_query = "A shelf" #>> change this
    query_tensor = text2embed3d(text_query)  # 1 x 3 tensor

    ptcd_path = '/mnt/c2d9b23a-b03e-4fdb-82ad-59f039ec9e3e/intern/King_Hang/OFFLINE-samslam_2/experiments/Replica/room0_0/params.npz'
    params = load_pointcloud(ptcd_path)
    le_s = params['pointwise_le'][:,0:3]
    le_m = params['pointwise_le'][:,3:6]
    le_l = params['pointwise_le'][:,6:9]

    # create sub-pointcloud
    indices_s = query_ptcd(query_tensor, le_s)
    indices_m = query_ptcd(query_tensor, le_m)
    indices_l = query_ptcd(query_tensor, le_l)

    indices = max([indices_s, indices_m, indices_l], key=len)
    
    # create splat
    means = params['means3D'][indices]
    scales = params['log_scales'][indices]
    rotations = params['unnorm_rotations'][indices]
    rgbs = params['rgb_colors'][indices]
    opacities = params['logit_opacities'][indices]

    work_path = config['workdir']
    run_name = config['run_name']
    ply_path = os.path.join(work_path, run_name, 'queries', "query_splat.ply")
    save_ply(ply_path, means, scales, rotations, rgbs, opacities) # save rgb