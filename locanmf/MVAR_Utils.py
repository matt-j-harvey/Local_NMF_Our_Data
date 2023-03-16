import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from skimage.transform import resize, rescale

from scipy import ndimage
import pathlib
import os

def create_image_from_data(data, indicies, image_height, image_width):
    template = np.zeros((image_height, image_width))
    data = np.nan_to_num(data)
    np.put(template, indicies, data)
    image = np.ndarray.reshape(template, (image_height, image_width))
    return image


def get_blue_black_cmap():
    cmap = LinearSegmentedColormap.from_list('mycmap', [
        (0, 0.87, 0.9, 1),
        (0, 0, 1, 1),
        (0, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 1, 0, 1),

    ])
    return cmap



def get_chunk_structure(chunk_size, array_size):
    number_of_chunks = int(np.ceil(array_size / chunk_size))
    remainder = array_size % chunk_size

    # Get Chunk Sizes
    chunk_sizes = []
    if remainder == 0:
        for x in range(number_of_chunks):
            chunk_sizes.append(chunk_size)

    else:
        for x in range(number_of_chunks - 1):
            chunk_sizes.append(chunk_size)
        chunk_sizes.append(remainder)

    # Get Chunk Starts
    chunk_starts = []
    chunk_start = 0
    for chunk_index in range(number_of_chunks):
        chunk_starts.append(chunk_size * chunk_index)

    # Get Chunk Stops
    chunk_stops = []
    chunk_stop = 0
    for chunk_index in range(number_of_chunks):
        chunk_stop += chunk_sizes[chunk_index]
        chunk_stops.append(chunk_stop)

    return number_of_chunks, chunk_sizes, chunk_starts, chunk_stops


def downsample_mask_further(indicies, image_height, image_width, downsample_size=100):
    template = np.zeros((image_height*image_width))
    template[indicies] = 1
    template = np.reshape(template, (image_height, image_width))
    template = resize(template, (downsample_size, downsample_size), anti_aliasing=True)
    template = np.reshape(template, (downsample_size * downsample_size))
    template = np.where(template > 0.5, 1, 0)
    template_indicies = np.nonzero(template)
    return template_indicies, downsample_size, downsample_size


def load_tight_mask():
    tight_mask_dict = np.load("Tight_Mask_Dict.npy", allow_pickle=True)[()]
    indicies = tight_mask_dict["indicies"]
    image_height = tight_mask_dict["image_height"]
    image_width = tight_mask_dict["image_width"]
    return indicies, image_height, image_width


def load_generous_mask(base_directory):

    generous_mask_dict = np.load(os.path.join(base_directory, "Downsampled_mask_dict.npy"), allow_pickle=True)[()]
    indicies = generous_mask_dict["indicies"]
    image_height = generous_mask_dict["image_height"]
    image_width = generous_mask_dict["image_width"]
    return indicies, image_height, image_width



def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)



def load_across_mice_alignment_dictionary(base_directory):

    # Get Root Directory
    base_directory_parts = pathlib.Path(base_directory)
    base_directory_parts = list(base_directory_parts.parts)
    root_directory = base_directory_parts[0]
    for subfolder in base_directory_parts[1:-1]:
        root_directory = os.path.join(root_directory, subfolder)

    print("Root Directory", root_directory)
    # Load Alignment Dictionary
    across_mouse_alignment_dictionary = np.load(os.path.join(root_directory, "Across_Mouse_Alignment_Dictionary.npy"), allow_pickle=True)[()]

    return across_mouse_alignment_dictionary




def transform_image(image, variable_dictionary, invert=False):

    # Settings
    background_size = 1000
    background_offset = 200
    origional_height, origional_width = np.shape(image)
    window_y_start = background_offset
    window_y_stop = window_y_start + origional_height
    window_x_start = background_offset
    window_x_stop = window_x_start + origional_width

    # Unpack Transformation Details
    angle = variable_dictionary['rotation']
    x_shift = variable_dictionary['x_shift']
    y_shift = variable_dictionary['y_shift']
    scale_factor = variable_dictionary['zoom']

    # Inverse
    if invert == True:
        angle = -1 * angle
        x_shift = -1 * x_shift
        y_shift = -1 * y_shift
        scale_factor = 1 - scale_factor
    else:
        scale_factor = 1 + scale_factor

    # Copy
    transformed_image = np.copy(image)

    # Scale
    transformed_image = rescale(transformed_image, scale_factor, anti_aliasing=False, preserve_range=True)

    # Rotate
    transformed_image = ndimage.rotate(transformed_image, angle, reshape=False, prefilter=True)

    # Translate
    background = np.zeros((background_size, background_size))
    new_height, new_width = np.shape(transformed_image)

    y_start = background_offset + y_shift
    y_stop = y_start + new_height

    x_start = background_offset + x_shift
    x_stop = x_start + new_width

    background[y_start:y_stop, x_start:x_stop] = transformed_image

    # Get Normal Sized Window
    transformed_image = background[window_y_start:window_y_stop, window_x_start:window_x_stop]

    return transformed_image
