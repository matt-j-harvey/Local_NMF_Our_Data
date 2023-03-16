import os

"""
number_of_threads = 35
os.environ["OMP_NUM_THREADS"] = str(number_of_threads) # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = str(number_of_threads) # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = str(number_of_threads) # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = str(number_of_threads) # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = str(number_of_threads) # export NUMEXPR_NUM_THREADS=1
"""

from sklearn.linear_model import LinearRegression, Ridge
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from scipy.stats import zscore
from scipy.cluster.hierarchy import ward, dendrogram, leaves_list
from scipy.spatial.distance import pdist, cdist
from sklearn.decomposition import TruncatedSVD, NMF

import MVAR_Utils


from datetime import datetime
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from tqdm import tqdm
import os


def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)


def get_cross_validated_ridge_penalties(design_matrix, target_data, n_folds=5):

    # Get Selection Of Potential Ridge Penalties
    ridge_penalty_selection = np.logspace(start=-3, stop=5, base=10, num=9)

    # Create Cross Fold Object
    cross_fold_object = KFold(n_splits=n_folds, random_state=None, shuffle=False)

    penalty_error_matrix = []

    # Iterate Through Each Ridge Penalty
    for penalty in tqdm(ridge_penalty_selection, desc="Ridge Penalty CV"):
        error_list = []

        print("Penalty", penalty)
        # Enumerate Through Each Fold
        for i, (train_indices, test_indices) in enumerate(cross_fold_object.split(design_matrix)):

            # Get Training and Test Data
            x_train = design_matrix[train_indices]
            y_train = target_data[train_indices]
            x_test = design_matrix[test_indices]
            y_test = target_data[test_indices]

            # Create Model
            model = Ridge(alpha=penalty, solver='auto')

            # Fit Model
            model.fit(X=x_train, y=y_train)

            # Predict Data
            y_pred = model.predict(X=x_test)

            # Score Prediction
            fold_error = mean_squared_error(y_true=y_test, y_pred=y_pred, multioutput='raw_values')
            error_list.append(fold_error)

        # Get Average Error Across Folds
        error_list = np.array(error_list)
        mean_error = np.mean(error_list, axis=0)
        penalty_error_matrix.append(mean_error)


    # Return The Ridge Penalties Associated With The Smallest Error For Each Pixel
    penalty_error_matrix = np.array(penalty_error_matrix)

    penalty_error_matrix = np.transpose(penalty_error_matrix)
    ridge_coef_vector = []
    for pixel_errors in penalty_error_matrix:
        min_error = np.min(pixel_errors)
        min_index = list(pixel_errors).index(min_error)
        selected_ridge_penalty = ridge_penalty_selection[min_index]
        ridge_coef_vector.append(selected_ridge_penalty)

    ridge_coef_vector = np.array(ridge_coef_vector)
    return ridge_coef_vector




def sort_matrix(matrix):

    # Cluster Matrix
    Z = ward(pdist(matrix))

    # Get Dendogram Leaf Order
    new_order = leaves_list(Z)

    # Sorted Matrix
    sorted_matrix = matrix[:, new_order][new_order]

    return sorted_matrix



def moving_average(array, window=3):

    number_of_frames = np.shape(array)[0]
    smoothed_data = []

    for frame_index in tqdm(range(0, number_of_frames-window)):
        smoothed_data.append(np.mean(array[frame_index:frame_index + window], axis=0))

    smoothed_data = np.array(smoothed_data)
    return smoothed_data


def highcut_filter(X, w = 7.5, fs = 30.):
    from scipy.signal import butter, filtfilt
    b, a = butter(2,w/(fs/2.), btype='lowpass');
    return filtfilt(b, a, X, padlen=10000, axis=0)


def denoise_data(array):
    model = TruncatedSVD(n_components=100)
    transformed_data = model.fit_transform(array)
    array = model.inverse_transform(transformed_data)
    return array


def denoise_data_NMF(array):
    min_value = np.min(array)
    array = np.subtract(array, min_value)
    model = NMF(n_components=50)
    transformed_data = model.fit_transform(array)
    array = model.inverse_transform(transformed_data)
    array = np.add(array, min_value)
    return array

def view_sample(delta_f_matrix):

    indicies, image_height, image_width = MVAR_Utils.load_tight_mask()
    indicies, image_height, image_width = MVAR_Utils.downsample_mask_further(indicies, image_height, image_width)
    colourmap = MVAR_Utils.get_blue_black_cmap()

    plt.ion()

    for frame in delta_f_matrix:

        frame = MVAR_Utils.create_image_from_data(frame, indicies, image_height, image_width)
        plt.imshow(frame, vmin=-0.03, vmax=0.03, cmap=colourmap)
        plt.draw()
        plt.pause(0.1)
        plt.clf()


def view_sample_delta(delta_f_matrix):

    indicies, image_height, image_width = MVAR_Utils.load_tight_mask()
    indicies, image_height, image_width = MVAR_Utils.downsample_mask_further(indicies, image_height, image_width)
    colourmap = MVAR_Utils.get_blue_black_cmap()

    plt.ion()

    number_of_frames = np.shape(delta_f_matrix)[0]

    for frame_index in range(number_of_frames-1):

        frame = np.subtract(delta_f_matrix[frame_index+1], delta_f_matrix[frame_index])
        frame = MVAR_Utils.create_image_from_data(frame, indicies, image_height, image_width)
        plt.imshow(frame, vmin=-0.03, vmax=0.03, cmap=colourmap)
        plt.draw()
        plt.pause(0.5)
        plt.clf()




def lowcut_filter(X, w = 0.0033, fs = 30.):
    from scipy.signal import butter, filtfilt
    b, a = butter(2,w/(fs/2.), btype='highpass');
    return filtfilt(b, a, X, padlen=10000, axis=0)


def visualise_ridge_penalties(ridge_penalties):
    indicies, image_height, image_width = MVAR_Utils.load_tight_mask()
    indicies, image_height, image_width = MVAR_Utils.downsample_mask_further(indicies, image_height, image_width)
    ridge_map = MVAR_Utils.create_image_from_data(ridge_penalties, indicies, image_height, image_width)
    plt.imshow(ridge_map)
    plt.show()


def perform_minimum_mvar_cv(base_directory):

    # Load Delta F
    delta_f_matrix = np.load(os.path.join(base_directory, "Delta_F_Matrix_100_by_100_SVD_200.npy"))
    delta_f_matrix = delta_f_matrix[3000:]
    print("Dleta F matrix", np.shape(delta_f_matrix))

    #delta_f_matrix = highcut_filter(delta_f_matrix, w=2)
    #delta_f_matrix = lowcut_filter(delta_f_matrix, w=0.1)


    # Subtract Mean
    #delta_f_mean = np.mean(delta_f_matrix, axis=0)
    #delta_f_matrix = np.subtract(delta_f_matrix, delta_f_mean)

    #view_sample(delta_f_matrix)
    #view_sample_delta(delta_f_matrix)

    # Denoise
    #delta_f_matrix = denoise_data(delta_f_matrix)

    # Smooth
    n_window = 3
    delta_f_matrix = moving_average(delta_f_matrix, window=n_window)

    # Z Score
    #delta_f_matrix = zscore(delta_f_matrix, axis=0)
    #delta_f_matrix = np.nan_to_num(delta_f_matrix)
    print("Dleta F matrix", np.shape(delta_f_matrix))

    shift = 1
    predictor_matrix = delta_f_matrix[0:-shift]
    output_matrix = delta_f_matrix[shift:]


    # Get Ridge Penalties
    #ridge_penalties = get_cross_validated_ridge_penalties(design_matrix=predictor_matrix, target_data=output_matrix)
    #np.save(r"/media/matthew/External_Harddrive_3/Angus_MVAR_Data/ridge_penalties.npy", ridge_penalties)

    ridge_penalties = np.load(r"/media/matthew/External_Harddrive_3/Angus_MVAR_Data/ridge_penalties.npy")

    visualise_ridge_penalties(ridge_penalties)


    predictor_matrix = predictor_matrix[0:10000]
    output_matrix = output_matrix[0:10000]

    # Fit Full Model
    model = Ridge(alpha=ridge_penalties)
    print("Starting TO Fit", datetime.now())
    model.fit(X=predictor_matrix, y=output_matrix)
    print("Finnished Fitting", datetime.now())

    coefs = model.coef_
    np.save(r"/media/matthew/External_Harddrive_3/Angus_MVAR_Data/Minimum_MVAR_Matrix.npy", coefs)

    magnitude = np.percentile(np.abs(coefs), 99)
    colourmap = MVAR_Utils.get_blue_black_cmap()

    plt.imshow(coefs, vmin=-magnitude, vmax=magnitude, cmap=colourmap)
    plt.show()


    sorted_coefs = sort_matrix(coefs)
    plt.imshow(sorted_coefs, vmin=-magnitude, vmax=magnitude, cmap=colourmap)
    plt.show()



def view_components():
    components = np.load("/media/matthew/External_Harddrive_3/Angus_MVAR_Data/SVD_Components.npy")

    for component in components:
        indicies, image_height, image_width = MVAR_Utils.load_tight_mask()
        indicies, image_height, image_width = MVAR_Utils.downsample_mask_further(indicies, image_height, image_width)
        map = MVAR_Utils.create_image_from_data(component, indicies, image_height, image_width)
        plt.imshow(map)
        plt.show()



class mvar_model():

    def __init__(self, ridge_penalties):
        number_of_regressors = np.shape(ridge_penalties)[0]
        self.Tikhonov = np.eye(number_of_regressors)
        self.Tikhonov = np.multiply(ridge_penalties, self.Tikhonov)

    def fit(self, design_matrix, delta_f_matrix):
        self.MVAR_parameters = np.linalg.solve(design_matrix.T @ design_matrix + self.Tikhonov.T @ self.Tikhonov, design_matrix.T @ delta_f_matrix.T)



def fit_model_pixelwise(design_marix, delta_f_matrix, ridge_penalty_matrix):

    number_of_pixels = np.shape(delta_f_matrix)[1]

    weight_matrix = np.zeros((number_of_pixels, number_of_pixels))

    for pixel_index in tqdm(range(number_of_pixels), desc="Fitting MVAR"):
        pixel_distance_penalties = ridge_penalty_matrix[pixel_index]
        model = mvar_model(pixel_distance_penalties)
        model.fit(design_marix, delta_f_matrix[:, pixel_index])
        weight_matrix[pixel_index] = model.MVAR_parameters

    return weight_matrix


"""
def runpar(f, design_marix, delta_f_matrix, ridge_penalty_matrix, ):

    '''
    res = runpar(function,          # function to execute
                 data,              # data to be passed to the function
                 nprocesses = None, # defaults to the number of cores on the machine
                 **kwargs)          # additional arguments passed to the function (dictionary)
    '''

    with Pool(initializer = parinit, processes=10) as pool:
        res = pool.map(partial(f,**kwargs),X)
    pool.join()
    return res
"""

def view_covariance_matrix(predictor_matrix, output_matrix):

    number_of_pixels = np.shape(predictor_matrix)[1]

    # Get Change Covariance
    output_matrix = np.subtract(output_matrix, predictor_matrix)


    covariance_matrix = np.cov(m=output_matrix, y=predictor_matrix, rowvar=False)
    print("Covariance matrix shape", np.shape(covariance_matrix))
    covariance_matrix = covariance_matrix[number_of_pixels:2 * number_of_pixels, number_of_pixels:2 * number_of_pixels]
    np.save("/media/matthew/External_Harddrive_3/Angus_MVAR_Data/Covariance_Matrix.npy", covariance_matrix)
    plt.title("Covar")
    plt.imshow(covariance_matrix)
    plt.show()


def fit_minimum_mvar_multiple_folds(design_matrix, target_data, n_folds):

    # Create Cross Fold Object
    cross_fold_object = KFold(n_splits=n_folds, random_state=None, shuffle=False)

    # Enumerate Through Each Fold
    coef_list = []
    for i, (train_indices, test_indices) in enumerate(cross_fold_object.split(design_matrix)):
        print("FOld: ", i)

        # Get Training and Test Data
        x_train = design_matrix[train_indices]
        y_train = target_data[train_indices]
        x_test = design_matrix[test_indices]
        y_test = target_data[test_indices]

        # Create Model
        model = Ridge(alpha=0.1, solver='auto')

        # Fit Model
        model.fit(X=x_train, y=y_train)

        coef_list.append(model.coef_)

    # Get Average Error Across Folds
    coef_list = np.array(coef_list)
    mean_coefs = np.mean(coef_list, axis=0)

    return mean_coefs



def reconstruct_pixelwise_connectivity_matrix(base_directory):

    # Load Spatial Components
    spatial_components = np.load(os.path.join(base_directory, "Local_NMF", "Spatial_Components.npy"))
    image_height, image_with, components = np.shape(spatial_components)
    spatial_components = np.reshape(spatial_components, (image_height * image_with, components))

    # load Regression Coefs
    regression_coefs = np.load(os.path.join(base_directory, "Local_NMF", "MVAR_Coefs.npy"))

    print("spatial ceofs", np.shape(spatial_components))
    print("regression coefs", np.shape(regression_coefs))

    inputs_reconstrcucted = np.dot(spatial_components, regression_coefs)
    print("inputs recon", np.shape(inputs_reconstrcucted))

    full_coefs = np.dot(inputs_reconstrcucted, spatial_components.T)
    print(" full coefs", np.shape(full_coefs))

    np.save(os.path.join(base_directory, "Local_NMF", "pixelwise_connectivity_matrix.npy"), full_coefs)


def perform_minimum_mvar(base_directory):

    # Load Temporal Components F
    temporal_components = np.load(os.path.join(base_directory, "Local_NMF", "Temporal_Components.npy"))
    temporal_components = np.transpose(temporal_components)

    temporal_components = moving_average(temporal_components)

    print("Temporal components", np.shape(temporal_components))
    shift = 1
    predictor_matrix = temporal_components[0:-shift]
    output_matrix = temporal_components[shift:]

    ridge_penalties = get_cross_validated_ridge_penalties(predictor_matrix, output_matrix)

    # Fit Full Model
    model = Ridge(alpha=ridge_penalties)
    model.fit(X=predictor_matrix, y=output_matrix)
    coefs = model.coef_


    np.save(os.path.join(base_directory, "Local_NMF", "MVAR_Coefs.npy"), coefs)

    plt.imshow(coefs)
    plt.show()


session_list = [
    r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_02_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_04_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_06_Switching_Imaging"]

for base_directory in session_list:
    perform_minimum_mvar(base_directory)
    reconstruct_pixelwise_connectivity_matrix(base_directory)