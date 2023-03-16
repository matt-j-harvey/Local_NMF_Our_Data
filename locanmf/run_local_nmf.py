import sys
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import torch
import time
import warnings
warnings.filterwarnings("ignore")

import LocaNMF
import local_nmf_video


def prepreprocess_data(V, U, brainmask, atlas, device, min_pixels, rank_range):

    q, r = np.linalg.qr(V.T)
    video_mats = (np.copy(U[brainmask]), r.T)

    region_mats = LocaNMF.extract_region_metadata(brainmask, atlas, min_size=min_pixels)
    region_metadata = local_nmf_video.RegionMetadata(region_mats[0].shape[0], region_mats[0].shape[1:], device=device)
    region_metadata.set(torch.from_numpy(region_mats[0].astype(np.uint8)), torch.from_numpy(region_mats[1]), torch.from_numpy(region_mats[2].astype(np.int64)))

    region_videos = LocaNMF.factor_region_videos(video_mats, region_mats[0], rank_range[1], device=device)

    low_rank_video = local_nmf_video.LowRankVideo((int(np.sum(brainmask)),) + video_mats[1].shape, device=device)
    low_rank_video.set(torch.from_numpy(video_mats[0].T), torch.from_numpy(video_mats[1]))

    return low_rank_video, region_metadata, region_videos, q




def run_local_nmf(U, V, atlas, brainmask):

    """
    # U - Spatial Components (Image_Height, Image_Width, N_Components)
    # V - Temporal Components (N_Components, N_Timepoints)
    # Atlas - (Image_Height, Image_Width) - Each Pixel Labelled accoridng to its corresponding region
    # Brainmask - (Image_Height, Image_Width) - True if in brain, false otherwise
    """

    # User Defined Parameters
    device = "cpu"
    minrank = 1;
    maxrank = 10;  # rank = how many components per brain region. Set maxrank to around 10 for regular dataset.
    rank_range = (minrank, maxrank, 1)
    min_pixels = 50  # minimum number of pixels in Allen map for it to be considered a brain region
    loc_thresh = 70  # Localization threshold, i.e. percentage of area restricted to be inside the 'Allen boundary'
    r2_thresh = 0.99  # Fraction of variance in the data to capture with LocaNMF
    nonnegative_temporal = False  # Do you want nonnegative temporal components? The data itself should also be nonnegative in this case.

    # Preprocress Data
    low_rank_video, region_metadata, region_videos, q = prepreprocess_data(V, U, brainmask, atlas, device, min_pixels, rank_range)

    # Run Local NMF
    locanmf_comps = LocaNMF.rank_linesearch(low_rank_video,
                                            region_metadata,
                                            region_videos,
                                            maxiter_rank=maxrank - minrank + 1,
                                            maxiter_lambda=40,
                                            maxiter_hals=20,
                                            lambda_step=1.35,
                                            lambda_init=1e-6,
                                            loc_thresh=loc_thresh,
                                            r2_thresh=r2_thresh,
                                            rank_range=rank_range,
                                            nnt=nonnegative_temporal,
                                            verbose=[True, False, False],
                                            sample_prop=(1, 1),
                                            device=device)

    A = locanmf_comps.spatial.data.cpu().numpy().T
    A_reshape = np.zeros((brainmask.shape[0], brainmask.shape[1], A.shape[1]));
    A_reshape.fill(np.nan)
    A_reshape[brainmask, :] = A


    temporal_components = np.matmul(q, locanmf_comps.temporal.data.cpu().numpy().T).T

    return A_reshape, temporal_components



"""
# Load Data
data_folder = r"/home/matthew/Documents/Github_Code_Clean/Local_NMF/demo/data/simulated"

# Load Atlas
atlas = sio.loadmat(os.path.join(data_folder, "atlas.mat"))['atlas'].astype(float)

# load Compressed Data
arrays = sio.loadmat(os.path.join(data_folder, 'Vc_Uc.mat'))
V = arrays['Vc']
U = arrays['Uc']
brainmask = arrays['brainmask'] == 1
del arrays

print("Brainmask Shape", np.shape(brainmask))


local_nmf_components = run_local_nmf(U, V, atlas, brainmask)


num_components = np.shape(A_reshape)[2]

for component_index in range(num_components):
    plt.imshow(A_reshape[:, :, component_index])
    plt.show()
"""

