import run_local_nmf
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import canny
from skimage.morphology import binary_dilation
from matplotlib.gridspec import GridSpec
from skimage.transform import downscale_local_mean, resize
from scipy import ndimage

from Widefield_Utils import widefield_utils



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


def load_mask():

    indicies, image_height, image_width = widefield_utils.load_tight_mask()
    indicies, image_height, image_width = widefield_utils.downsample_mask_further(indicies, image_height, image_width)

    template = np.zeros(image_height * image_width)
    template[indicies] = 1
    template = np.reshape(template, (image_height, image_width))
    template = np.ndarray.astype(template, bool)
    return template



def load_atlas():

    # Load Pixel Dict
    #region_dict = np.load("/home/matthew/Documents/Github_Code_Clean/Widefield_Analysis/Files/Allen_Region_Dict.npy", allow_pickle=True)[()]
    #atlas = region_dict['pixel_labels']
    atlas = np.load(r"/home/matthew/Documents/Allen_Atlas_Templates/churchland_atlas.npy")

    plt.imshow(atlas, cmap='prism')
    plt.show()

    # Load Atlas Dict
    #atlas_alignment_dict = np.load("/home/matthew/Documents/Github_Code_Clean/Widefield_Analysis/Files/Atlas_Alignment_Dictionary.npy", allow_pickle=True)[()]
    #atlas_alignment_dict = np.load("/home/matthew/Documents/Allen_Atlas_Templates/Churchland_Atlas_Alignment_Dictionary.npy", allow_pickle=True)[()]
    atlas_alignment_dict = np.load("/home/matthew/Documents/Allen_Atlas_Templates/Churchland_Atlas_Alignment_Dictionary_Combined_Sign_Map.npy", allow_pickle=True)[()]

    # Align Atlas
    atlas = transform_atlas_regions(atlas, atlas_alignment_dict)


    return atlas


def align_u(base_directory, U):

    # Load Within Mouse Alignment Dict
    within_mouse_alignment_dict = np.load(os.path.join(base_directory, "Within_Mouse_Alignment_Dictionary.npy"), allow_pickle=True)[()]

    # Load Across Mouse Alignment Dict
    across_mouse_alignment_dict = widefield_utils.load_across_mice_alignment_dictionary(base_directory)

    image_height, image_width, number_of_components = np.shape(U)
    aligned_components_list = np.zeros((100, 100, number_of_components))

    for component_index in range(number_of_components):
        component = U[:, :, component_index]
        component = widefield_utils.transform_image(component, within_mouse_alignment_dict)
        component = widefield_utils.transform_image(component, across_mouse_alignment_dict)

        component = component[0:300, 0:300]
        component = downscale_local_mean(component, (3, 3))

        aligned_components_list[:, :, component_index] = component

    return aligned_components_list


def view_aligned_u(u, atlas):

    atlas_edges = canny(atlas)
    edge_indicies = np.nonzero(atlas_edges)

    image_height, image_width, number_of_components = np.shape(u)
    for component_index in range(number_of_components):

        component = u[:, :, component_index]
        component[edge_indicies] = np.max(component)
        plt.imshow(component)
        plt.show()


def view_components(components, nmf_directory):

    num_components = np.shape(components)[2]

    # View Combined Mapping
    x, y = widefield_utils.get_best_grid(num_components)
    figure_1 = plt.figure()
    for component_index in range(num_components):
        axis_1 = figure_1.add_subplot(x, y, component_index + 1)
        axis_1.imshow(components[:, :, component_index])
        axis_1.axis('off')
    plt.show()

    # Plot Individual Comps
    save_directory = os.path.join(nmf_directory, "Spatial_Components")
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    for component_index in range(num_components):
        plt.imshow(components[:, :, component_index])
        plt.savefig(os.path.join(save_directory, str(component_index).zfill(3) + ".png"))
        plt.close()



def reconstruct_data(spatial_components, temporal_components, sample_size=10000):

    print("Spatial components", np.shape(spatial_components))
    print("Temporal components", np.shape(temporal_components))
    data = np.dot(spatial_components, temporal_components[:, 0:sample_size])

    print("Data Shape", np.shape(data))

    colourmap = widefield_utils.get_musall_cmap()

    plt.ion()
    for frame_index in range(sample_size):
        plt.imshow(data[:, :, frame_index], cmap=colourmap, vmin=-0.05, vmax=0.05)
        plt.draw()
        plt.pause(0.1)
        plt.clf()


def run_local_nmf_pipeline(base_directory, early_cutoff=3000):

    # Load Atlas
    atlas = load_atlas()
    atlas = downscale_atlas(atlas)

    # Load Mask
    mask = load_mask()

    # Load Data
    u = np.load(os.path.join(base_directory, "Churchland_Preprocessing", "U.npy"))
    v = np.load(os.path.join(base_directory, "Churchland_Preprocessing", "Corrected_SVT.npy"))
    v = v[:, early_cutoff:]

    # align U
    u = align_u(base_directory, u)
    # view_aligned_u(u, atlas)

    # Perform NMF
    spatial_components, temporal_components = run_local_nmf.run_local_nmf(u, v, atlas, mask)

    # Create Save Directory
    save_directory = os.path.join(base_directory, "Local_NMF")
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

        # View Components
    view_components(spatial_components, save_directory)
    print("Temporal Components Shape", np.shape(temporal_components))

    # Add Earlycutoff Back
    n_temporal_components = np.shape(temporal_components)[0]
    zero_padding = np.zeros((n_temporal_components, early_cutoff))
    print("Temporal Components Pre Padding Shape", np.shape(temporal_components))
    padded_temporal_components = np.hstack([zero_padding, temporal_components])
    print("Temporal Components Post Padding Shape", np.shape(padded_temporal_components))

    np.save(os.path.join(save_directory, "Spatial_Components.npy"), spatial_components)
    np.save(os.path.join(save_directory, "Temporal_Components.npy"), padded_temporal_components)

    #reconstruct_data(spatial_components, temporal_components)



session_list = [r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_02_Switching_Imaging",
                r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_04_Switching_Imaging",
                r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_06_Switching_Imaging"]

session_list = [
    r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_14_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_20_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_22_Switching_Imaging",
]

for base_directory in session_list:
    run_local_nmf_pipeline(base_directory)
