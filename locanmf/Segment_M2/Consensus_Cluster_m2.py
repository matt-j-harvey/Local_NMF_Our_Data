import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import ward, dendrogram, leaves_list
from scipy.spatial.distance import pdist, cdist
from skimage.feature import canny

from tqdm import tqdm

from Widefield_Utils import widefield_utils


def sort_matrix(matrix):

    # Cluster Matrix
    Z = ward(pdist(matrix))

    # Get Dendogram Leaf Order
    new_order = leaves_list(Z)

    # Sorted Matrix
    sorted_matrix = matrix[:, new_order][new_order]

    return sorted_matrix


def create_consensus_affinity_matrix(labels_list):
    n_pixels = np.shape(labels_list[0])[0]

    affinity_matrix = np.zeros((n_pixels, n_pixels))

    for session in tqdm(labels_list):
        for pixel_index_1 in range(n_pixels):
            for pixel_index_2 in range(n_pixels):

                if session[pixel_index_1] == session[pixel_index_2]:
                    affinity_matrix[pixel_index_1, pixel_index_2] += 1

    return affinity_matrix

control_transition_sessions_no_root_flat = [

    r"NXAK4.1B/2021_04_02_Transition_Imaging",
    r"NXAK4.1B/2021_04_08_Transition_Imaging",
    r"NXAK4.1B/2021_04_10_Transition_Imaging",

    r"NXAK7.1B/2021_03_23_Transition_Imaging",
    r"NXAK7.1B/2021_03_31_Transition_Imaging",
    r"NXAK7.1B/2021_04_02_Transition_Imaging",

    r"NXAK14.1A/2021_06_13_Transition_Imaging",
    r"NXAK14.1A/2021_06_15_Transition_Imaging",
    r"NXAK14.1A/2021_06_17_Transition_Imaging",

    r"NXAK22.1A/2021_10_29_Transition_Imaging",
    r"NXAK22.1A/2021_11_03_Transition_Imaging",
    r"NXAK22.1A/2021_11_05_Transition_Imaging",

]


save_root = r"/home/matthew/Documents/M2_Segmentation"


# Load Labels
label_list = []
for base_directory in tqdm(control_transition_sessions_no_root_flat):
   labels = np.load(os.path.join(save_root, base_directory, "M2_Labels.npy"))
   label_list.append(labels)

# Cluster
affinity_matrix = create_consensus_affinity_matrix(label_list)
affinity_matrix = np.divide(affinity_matrix, len(label_list))

#affinity_matrix = sort_matrix(affinity_matrix)

# print(np.shape(labels))
plt.imshow(affinity_matrix)
plt.show()

distance_matrix = 1 - affinity_matrix
model = AgglomerativeClustering(affinity='precomputed', distance_threshold=0.5, n_clusters=None, linkage='average')
model.fit(distance_matrix)
labels = model.labels_
labels = np.add(labels, 1)



# View Clustering
m2_indicies = np.load(os.path.join(save_root, "m2_indicies_100.npy"))
indicies, image_height, image_width = widefield_utils.load_tight_mask()
indicies, image_height, image_width = widefield_utils.downsample_mask_further(indicies, image_height, image_width)
n_pixels = np.shape(indicies)[1]
full_vector = np.zeros(n_pixels)
full_vector[m2_indicies] = labels
cluster_mapping = widefield_utils.create_image_from_data(full_vector, indicies, image_height, image_width)
plt.imshow(cluster_mapping)
plt.show()


# Reinsert Back
downsampled_churchland_atlas = np.load(os.path.join(save_root, "Downsampled_Churchland_Atlas.npy"))
plt.imshow(downsampled_churchland_atlas)
plt.show()

max_existing_label = np.max(downsampled_churchland_atlas)

new_atlas = np.copy(downsampled_churchland_atlas)
new_atlas = np.where(cluster_mapping == 1, max_existing_label + 1, new_atlas)
new_atlas = np.where(cluster_mapping == 2, max_existing_label + 2, new_atlas)
plt.imshow(new_atlas)
plt.show()

# Reflect
left_half = new_atlas[:, 0:49]
right_half = np.flip(left_half, axis=1)
right_half = np.multiply(right_half, -1)

full_map = np.zeros((image_height, image_width))
full_map[:, 0:49] = left_half
full_map[:, 49:98] = right_half
plt.imshow(full_map, cmap='prism')
plt.show()

edges = canny(full_map)
plt.imshow(edges)
plt.show()

np.save(os.path.join(save_root, "Downsampled_Churchland_Atlas_Split_M2.npy"), full_map)