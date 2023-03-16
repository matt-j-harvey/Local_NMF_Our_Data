import numpy as np
from skimage.transform import resize

from Widefield_Utils import widefield_utils


def create_index_map( indicies, image_height, image_width):

    index_map = np.zeros(image_height * image_width)
    index_map[indicies] = list(range(np.shape(indicies)[1]))
    index_map = np.reshape(index_map, (image_height, image_width))

    """
    plt.title("Index map")
    plt.imshow(index_map)
    plt.show()
    """
    return index_map



def get_roi_pixels(roi_name):

    # Load Pixel Dict
    region_dict = np.load("/home/matthew/Documents/Github_Code_Clean/Widefield_Analysis/Files/Allen_Region_Dict.npy", allow_pickle=True)[()]
    pixel_labels = region_dict['pixel_labels']
    print("region_dict")
    print(region_dict)

    # Load Atlas Dict
    atlas_alignment_dict = np.load("/home/matthew/Documents/Github_Code_Clean/Widefield_Analysis/Files/Atlas_Alignment_Dictionary.npy", allow_pickle=True)[()]

    # Transform Atlas
    pixel_labels = widefield_utils.transform_atlas_regions(pixel_labels, atlas_alignment_dict)

    # Get Selected ROI Mask
    selected_roi_label = region_dict[roi_name]
    roi_mask = np.where(pixel_labels == selected_roi_label, 1, 0)

    # Downsample To 100
    downsample_size = 100
    roi_mask = resize(roi_mask, (downsample_size, downsample_size), anti_aliasing=True, preserve_range=True)
    roi_mask = np.around(roi_mask, decimals=0)

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


def get_pooled_roi_from_list(roi_list):

    pooled_roi_pixels = []
    for roi in roi_list:
        roi_pixel_indicies = get_roi_pixels(roi)
        for index in roi_pixel_indicies:
            pooled_roi_pixels.append(index)

    return pooled_roi_pixels


