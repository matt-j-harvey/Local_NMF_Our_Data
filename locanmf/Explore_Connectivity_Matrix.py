import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

import h5py
import tables
from scipy import signal, ndimage, stats
from sklearn.linear_model import LinearRegression
from skimage.morphology import white_tophat
from sklearn.preprocessing import StandardScaler
from skimage.transform import rescale, resize
from PIL import Image
import os
import cv2
import datetime

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import pyqtgraph
import sys

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.decomposition import FactorAnalysis, TruncatedSVD, FastICA, PCA, NMF
from sklearn.metrics import accuracy_score
from sklearn.cluster import MiniBatchKMeans, SpectralClustering, DBSCAN, AffinityPropagation
from sklearn.mixture import GaussianMixture
import networkx as nx
import cv2
from matplotlib.pyplot import cm
from matplotlib.colors import to_rgb
import os
from scipy.cluster.hierarchy import ward, dendrogram, leaves_list
from scipy.spatial.distance import pdist
from matplotlib import pyplot as plt
import random

from sklearn.decomposition import PCA

pyqtgraph.setConfigOptions(imageAxisOrder='row-major')

from Widefield_Utils import widefield_utils

class correlation_explorer(QWidget):

    def __init__(self, correlation_matrix, image_height, image_width, parent=None):
        super(correlation_explorer, self).__init__(parent)

        # Setup Window
        self.setWindowTitle("Correlation Modulation")
        self.setGeometry(0, 0, 1000, 500)

        # Create Variable Holders
        self.correlation_matrix = correlation_matrix
        self.image_height = image_height
        self.image_width = image_width

        colour_list = [[0, 0.87, 0.9, 1],
                        [0, 0, 1, 1],
                        [0, 0, 0, 1],
                        [1, 0, 0, 1],
                        [1, 1, 0, 1],]
        colour_list = np.array(colour_list)
        colour_list = np.multiply(colour_list, 255)

        value_list = np.linspace(0, 1, num=len(colour_list))
        print("Valye list", value_list)
        self.colourmap = pyqtgraph.ColorMap(pos=value_list, color=colour_list)

        # Create Display Views
        self.row_correlation_map_display_view, self.row_correlation_map_display_widget = self.create_display_widget()
        self.column_correlation_map_display_view, self.column_correlation_map_display_widget = self.create_display_widget()

        #self.row_correlation_map_display_view.setImage(MVAR_Utils.create_image_from_data(self.correlation_matrix[0], indicies, image_height, image_width))
        #self.column_correlation_map_display_view.setImage(MVAR_Utils.create_image_from_data(self.correlation_matrix[:, 0], indicies, image_height, image_width))

        # Create Display View Labels
        self.row_display_view_label = QLabel("Pixel Inputs")
        self.column_display_view_label = QLabel("Pixel Outputs")

        # Create Layout
        self.layout = QGridLayout()
        self.setLayout(self.layout)
        self.layout.addWidget(self.row_display_view_label, 0, 0, 1, 1)
        self.layout.addWidget(self.column_display_view_label, 0, 1, 1, 1)

        self.layout.addWidget(self.row_correlation_map_display_widget,      1, 0, 1, 1)
        self.layout.addWidget(self.column_correlation_map_display_widget,   1, 1, 1, 1)

    def create_index_map(self, indicies, image_height, image_width):
        index_map = np.zeros(image_height * image_width)
        index_list = list(range(np.shape(indicies)[1]))
        index_map[indicies] = index_list
        index_map = np.reshape(index_map, (image_height, image_width))
        return index_map

    def getPixel(self, pos, imageview):
        pos = imageview.getImageItem().mapFromScene(pos)
        y = np.clip(int(pos.y()), a_min=0, a_max=self.image_height - 1)
        x = np.clip(int(pos.x()), a_min=0, a_max=self.image_width - 1)

        pixel_index = y * image_height + x

        # Get Connectivity Vectors
        row_modulation = self.correlation_matrix[pixel_index]
        column_modulation = self.correlation_matrix[:, pixel_index]

        # Reconstruct Into 2D
        row_modulation_image = np.reshape(row_modulation, (self.image_height, self.image_width))
        column_modulation_image = np.reshape(column_modulation, (self.image_height, self.image_width))

        # Display These Images
        self.row_correlation_map_display_view.setImage(row_modulation_image)
        self.column_correlation_map_display_view.setImage(column_modulation_image)

        # Scale Colourmaps
        self.row_modulation_magnitude = np.max(np.abs(row_modulation)) #* 0.1
        self.column_modulation_magnitude = np.max(np.abs(column_modulation)) #* 0.1

        #self.row_modulation_magnitude = np.max(np.abs(self.correlation_matrix))
        #self.column_modulation_magnitude = np.max(np.abs(self.correlation_matrix))

        self.row_correlation_map_display_view.setLevels(-self.row_modulation_magnitude, self.row_modulation_magnitude)
        self.column_correlation_map_display_view.setLevels(-self.column_modulation_magnitude, self.column_modulation_magnitude)

        self.row_correlation_map_display_view.setColorMap(self.colourmap)
        self.column_correlation_map_display_view.setColorMap(self.colourmap)


    def create_display_widget(self):

        # Create Figures
        display_view_widget = QWidget()
        display_view_widget_layout = QGridLayout()
        display_view = pyqtgraph.ImageView()
        # display_view.setColorMap(self.colour_map)
        display_view.ui.histogram.hide()
        display_view.ui.roiBtn.hide()
        display_view.ui.menuBtn.hide()
        display_view_widget_layout.addWidget(display_view, 0, 0)
        display_view_widget.setLayout(display_view_widget_layout)
        # display_view_widget.setMinimumWidth(800)
        # display_view_widget.setMinimumHeight(800)

        display_view.getView().scene().sigMouseMoved.connect(lambda pos: self.getPixel(pos, display_view))

        return display_view, display_view_widget



def downsample_mask_further(indicies, image_height, image_width, downsample_size=100):
    template = np.zeros((image_height*image_width))
    template[indicies] = 1
    template = np.reshape(template, (image_height, image_width))
    template = resize(template, (downsample_size, downsample_size), anti_aliasing=True)
    template = np.reshape(template, (downsample_size * downsample_size))
    template = np.where(template > 0.5, 1, 0)
    template_indicies = np.nonzero(template)
    return template_indicies, downsample_size, downsample_size



def load_average_matrix(session_list, model_name):

    connectivity_matrix_list = []
    for base_directory in session_list:
        connectivity_matrix = np.load(os.path.join(base_directory, "Ridge_Weights", model_name + "_Interaction_Weights.npy"))
        connectivity_matrix_list.append(connectivity_matrix)

    connectivity_matrix_list = np.array(connectivity_matrix_list)
    connectivity_matrix = np.mean(connectivity_matrix_list, axis=0)
    return connectivity_matrix


def get_mean_connectivity_matrix(session_list, context):

    matrix_list = []
    for session in session_list:
        connectivity_matrix = load_connectivity_matrix(session, context)
        matrix_list.append(connectivity_matrix)

    matrix_list = np.array(matrix_list)
    connectivity_matrix = np.mean(matrix_list, axis=0)
    return connectivity_matrix


def load_connectivity_matrix(session, context):

    if context == "Comparison":
        visual_regression_dict = np.load(os.path.join(session, "Full_Model_Multi_Session", "visual_regression_dict.npy"), allow_pickle=True)[()]
        visual_connectivity_matrix = visual_regression_dict["Interaction_Weights"]

        odour_regression_dict = np.load(os.path.join(session, "Full_Model_Multi_Session", "odour_regression_dict.npy"), allow_pickle=True)[()]
        odour_connectivity_matrix = odour_regression_dict["Interaction_Weights"]
        connectivity_matrix = np.subtract(visual_connectivity_matrix, odour_connectivity_matrix)

    else:
        regression_dict = np.load(os.path.join(session, "Full_Model_Multi_Session", context + "_regression_dict.npy"), allow_pickle=True)[()]
        connectivity_matrix = regression_dict["Interaction_Weights"]

    return connectivity_matrix



def load_connectivity_matrix_distance_penalty(session, context):

    if context == "Comparison":
        visual_regression_dict = np.load(os.path.join(session, "Full_Model_Multi_Session", "visual_regression_dict_locality_penalty.npy"), allow_pickle=True)[()]
        visual_connectivity_matrix = visual_regression_dict["Interaction_Weights"]

        odour_regression_dict = np.load(os.path.join(session, "Full_Model_Multi_Session", "odour_regression_dict_locality_penalty.npy"), allow_pickle=True)[()]
        odour_connectivity_matrix = odour_regression_dict["Interaction_Weights"]
        connectivity_matrix = np.subtract(visual_connectivity_matrix, odour_connectivity_matrix)

    else:
        regression_dict = np.load(os.path.join(session, "Full_Model_Multi_Session", context + "_regression_dict_locality_penalty.npy"), allow_pickle=True)[()]
        connectivity_matrix = regression_dict["Interaction_Weights"]

    return connectivity_matrix


def load_average_connectivity_matrix(connectivity_matrix_file_list):

    matrix_list = []
    for connectivity_matrix_file in connectivity_matrix_file_list:
        matrix = np.load(connectivity_matrix_file)
        #matrix = np.divide(matrix, np.max(np.abs(matrix)))
        matrix_list.append(matrix)

    mean_matrix = np.mean(np.array(matrix_list), axis=0)
    return mean_matrix


def get_average_connectivity_matrix(session_list):
    connectivity_matrix_list = []
    for base_directory in session_list:
        connectivity_matrix = np.load(os.path.join(base_directory, "Local_NMF", "pixelwise_connectivity_matrix.npy"))
        connectivity_matrix = np.nan_to_num(connectivity_matrix)
        connectivity_matrix_list.append(connectivity_matrix)

    mean_connectivity_matrix = np.mean(np.array(connectivity_matrix_list), axis=0)
    return mean_connectivity_matrix


if __name__ == '__main__':

    app = QApplication(sys.argv)


    # Load Matrix
    session_list = [
    r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_02_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_04_Switching_Imaging",
    r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_06_Switching_Imaging",
        ]

    #connectivity_matrix = np.load(os.path.join(base_directory, "Local_NMF", "pixelwise_connectivity_matrix.npy"))
    #connectivity_matrix = np.nan_to_num(connectivity_matrix)

    connectivity_matrix = get_average_connectivity_matrix(session_list)


    image_height = 100
    image_width = 100

    window = correlation_explorer(connectivity_matrix, image_height, image_width)
    window.showMaximized()

    app.exec_()



