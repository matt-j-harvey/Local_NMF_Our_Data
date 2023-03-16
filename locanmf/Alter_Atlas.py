import numpy as np
import matplotlib.pyplot as plt


def merge_atlas(atlas, marge_list):

    for group in merge_list:
        group_label = group[0]
        for label in group:
            atlas = np.where(atlas == label, group_label, atlas)

    return atlas



merge_list = [
    [261, 255, 249],        [-261, -255, -249],
	[43, 57, 71],           [-43, -57, -71],
	[21, 8, 198, 186],      [-21, -8, -198, -186],
	[29, 36],               [-29, -36],
	[64, 268, 275, 129],    [-64, -268, -275, -129],
	[136, 164],             [-136, -164],
    [143, 157],             [-143, -157],
    [282, 178, 171, 143],   [-282, -178, -171, -143],
    [92, 78],               [-92, -78]

]

atlas = np.load(r"/home/matthew/Documents/Allen_Atlas_Templates/churchland_atlas.npy")

atlas = merge_atlas(atlas, merge_list)
np.save(r"/home/matthew/Documents/Allen_Atlas_Templates/churchland_atlas_simplified_single_m2.npy", atlas)

plt.imshow(atlas, cmap='flag')
plt.show()