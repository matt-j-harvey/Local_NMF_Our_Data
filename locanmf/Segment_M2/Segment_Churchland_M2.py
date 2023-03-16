import run_local_nmf
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import canny
from skimage.morphology import binary_dilation
from matplotlib.gridspec import GridSpec
from skimage.transform import downscale_local_mean, resize
from scipy import ndimage
from sklearn.cluster import KMeans, AffinityPropagation
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt


import ROI_Quantification_Functions
from Widefield_Utils import widefield_utils

def downscale_atlas(atlas):

    atlas = atlas[0:300, 0:300]

    downsampled_atlas = np.zeros((100, 100))
    unique_regions = np.unique(atlas)

    for region in unique_regions:
        if region != 0:
            region_mask = np.where(atlas == region, 1, 0)
            region_mask = downscale_local_mean(region_mask, (3, 3))
            region_mask = np.where(region_mask > 0.5, 1, 0)
            region_mask = binary_dilation(region_mask)
            region_indicies = np.nonzero(region_mask)
            downsampled_atlas[region_indicies] = region


    left_half = downsampled_atlas[:, 0:52]
    right_half = left_half[:, 4:]
    right_half = np.flip(right_half, axis=1)
    right_half = np.multiply(right_half, -1)

    downsampled_atlas[:, 52:] = right_half

    downsampled_atlas = np.roll(downsampled_atlas, axis=1, shift=-3)
    downsampled_atlas = np.roll(downsampled_atlas, axis=0, shift=2)
    return downsampled_atlas


def transform_mask_or_atlas_300(image, variable_dictionary):

    image_height = 300
    image_width = 304

    # Unpack Dictionary
    angle = variable_dictionary['rotation']
    x_shift = variable_dictionary['x_shift']
    y_shift = variable_dictionary['y_shift']
    x_scale = variable_dictionary['x_scale']
    y_scale = variable_dictionary['y_scale']

    # Copy
    transformed_image = np.copy(image)
    transformed_image = np.ndarray.astype(transformed_image, float)

    # Scale
    original_height, original_width = np.shape(transformed_image)
    new_height = int(original_height * y_scale)
    new_width = int(original_width * x_scale)
    transformed_image = resize(transformed_image, (new_height, new_width), preserve_range=True, anti_aliasing=True)

    # Rotate
    transformed_image = ndimage.rotate(transformed_image, angle, reshape=False, prefilter=True)

    # Insert Into Background
    mask_height, mask_width = np.shape(transformed_image)
    centre_x = 200
    centre_y = 200
    background_array = np.zeros((1000, 1000))
    x_start = centre_x + x_shift
    x_stop = x_start + mask_width

    y_start = centre_y + y_shift
    y_stop = y_start + mask_height

    background_array[y_start:y_stop, x_start:x_stop] = transformed_image

    # Take Chunk
    transformed_image = background_array[centre_y:centre_y + image_height, centre_x:centre_x + image_width]

    # Rebinarize
    transformed_image = np.where(transformed_image > 0.5, 1, 0)

    return transformed_image


def transform_atlas_regions(image, variable_dictionary):

    unique_values = list(set(np.unique(image)))
    transformed_mask = np.zeros(np.shape(image))
    for value in unique_values:
        value_mask = np.where(image == value, 1, 0)
        value_mask = transform_mask_or_atlas_300(value_mask, variable_dictionary)
        value_mask = binary_dilation(value_mask)
        value_indicies = np.nonzero(value_mask)
        transformed_mask[value_indicies] = value
    return transformed_mask


def load_atlas():

    # Load Pixel Dict
    atlas = np.load(r"/home/matthew/Documents/Allen_Atlas_Templates/churchland_atlas_simplified_single_m2.npy")

    # Load Atlas Dict
    atlas_alignment_dict = np.load("/home/matthew/Documents/Allen_Atlas_Templates/Churchland_Atlas_Alignment_Dictionary_Combined_Sign_Map.npy", allow_pickle=True)[()]

    # Align Atlas
    atlas = transform_atlas_regions(atlas, atlas_alignment_dict)


    return atlas



def create_index_map( indicies, image_height, image_width):
    index_map = np.zeros(image_height * image_width)
    index_map[indicies] = list(range(np.shape(indicies)[1]))
    index_map = np.reshape(index_map, (image_height, image_width))
    return index_map



def get_m2_indicies(atlas):

    atlas = np.abs(atlas)
    roi_mask = np.where(atlas == 21, 1, 0)

    # Load Mask
    indicies, image_height, image_width = widefield_utils.load_tight_mask()
    indicies, image_height, image_width = widefield_utils.downsample_mask_further(indicies, image_height, image_width)

    # Create Index Map
    index_map = create_index_map(indicies, image_height, image_width)

    # Get ROI Indicies
    roi_world_indicies = np.nonzero(roi_mask)
    roi_pixel_indicies = index_map[roi_world_indicies]
    roi_pixel_indicies = np.array(roi_pixel_indicies, dtype=np.int)
    return roi_pixel_indicies





def functional_parcellation(data_root, base_directory, m2_indicies, save_directory):

    #Load delta F
    delta_f = np.load(os.path.join(data_root, base_directory, "Delta_F_Matrix_100_by_100_SVD.npy"))
    n_timepoints, n_pixels = np.shape(delta_f)

    print(np.shape(delta_f))

    # Get M2 Activity
    m2_activity = delta_f[:, m2_indicies]
    print("M2 Activity", np.shape(m2_activity))

    #get_elbow(np.transpose(m2_activity))

    kmeans = KMeans(n_clusters=3).fit(np.transpose(m2_activity))
    labels = kmeans.labels_

    #model = AffinityPropagation()
    #model.fit(np.transpose(m2_activity))
    #labels = model.labels_

    print("Labels Shape", np.shape(labels))
    full_vector = np.ones(n_pixels) * -1
    full_vector[m2_indicies] = labels


    # View Clustering
    indicies, image_height, image_width = widefield_utils.load_tight_mask()
    indicies, image_height, image_width = widefield_utils.downsample_mask_further(indicies, image_height, image_width)

    cluster_mapping = widefield_utils.create_image_from_data(full_vector, indicies, image_height, image_width)
    #plt.imshow(cluster_mapping)
    #plt.show()
    print("m2_pixels", np.shape(m2_activity))


    full_save_directory = os.path.join(save_directory, base_directory)
    if not os.path.exists(full_save_directory):
        os.makedirs(full_save_directory)

    np.save(os.path.join(full_save_directory, "M2_Labels.npy"), labels)






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

data_root = r"/media/matthew/Expansion/Control_Data"
save_root = r"/home/matthew/Documents/M2_Segmentation"

# Load Atlas
atlas = load_atlas()
atlas = downscale_atlas(atlas)
np.save(os.path.join(save_root, "Downsampled_Churchland_Atlas.npy"), atlas)
plt.imshow(atlas)
plt.show()

# Get M2 Indicies
m2_indicies = get_m2_indicies(atlas)
np.save(os.path.join(save_root, "m2_indicies_100.npy"), m2_indicies)

# Load Delta F
for base_directory in tqdm(control_transition_sessions_no_root_flat):
    functional_parcellation(data_root, base_directory, m2_indicies, save_root)