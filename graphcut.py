import numpy as np
import cv2
import random
import networkx as nx

# Ruiqi Yin - Final Project

class GraphCut(object):
    def __init__(self, left_img, right_img, ndisp, disparity_step=1, kernel=3):
        """
        left_img, right_img: input images stereo
        ndisp:   a conservative bound on the number of disparity levels;
               the stereo algorithm MAY utilize this bound and search from d = 0 .. ndisp-1
               from dataset
        disparity_step: for going through all disparities stride
        kernel: kernel size for calculating Data Energy

        """

        print('-------GraphCut Start-------')
        print(left_img.shape)
        len_shape = len(left_img.shape)
        if len_shape == 2:
            print('USE GRAYSCALE')
        else:
            print('USE RBG')

        # For DEBUG USE ONLY
        assert left_img.shape == right_img.shape

        # initialize the class variables
        print('initialize the class variables for the Graph')
        self.ndisp = ndisp
        height, width = self.shape_original = right_img.shape[:2]
        self.image_length = height * width
        # print('self.image_length', self.image_length)
        self.labels = np.arange(0, ndisp, disparity_step)   # Labels to compare with pred
        self.preds = np.random.choice(self.labels, size=self.shape_original, replace=True)
        self.preds = self.preds.flatten()
        self.kernel = kernel

        self.initialize_images(left_img, right_img)

        # Two thresholds for the energy functions
        self.E_Data_threshold, self.E_Smooth_threshold = 110, 10


    # Function to preprocess stereo images and initialze them as class vars
    def initialize_images(self, left_img, right_img):
        # Create borders for the image
        pad = int((self.kernel - 1) / 2)
        padded_left_img = cv2.copyMakeBorder(left_img, top=pad, bottom=pad, left=pad, right=pad,
                                             borderType=cv2.BORDER_REFLECT)
        padded_right_img = cv2.copyMakeBorder(right_img, top=pad, bottom=pad, left=pad, right=pad,
                                              borderType=cv2.BORDER_REFLECT)
        # Stack zeros horizontally to the padded images
        height = self.shape_original[0]
        zero_padding = np.zeros((height+2*pad, self.ndisp, 3))
        self.left_image = np.hstack((zero_padding, padded_left_img))
        self.right_image = np.hstack((zero_padding, padded_right_img))


    def calculate_energy_data(self, patch1, patch2):
        # TODO: Trying with different metrics
        metric = 'mean'
        if metric == 'diff':
        # 1. try with diff
            data_energy = (patch1 - patch2)**2
        elif metric == 'mean':
        # 2. try with mean
            data_energy = ((patch1 - patch2)**2).mean()
        else:
            data_energy = np.sqrt((patch1 - patch2)**2)

        return int(min(data_energy, self.E_Data_threshold))


    def calculate_energy_smooth(self, val1, val2):
        # TODO: Trying with different values, and metric
        metric = 'square'
        if metric == 'square':
            diff = (val1 - val2)**2
        else:
            diff = abs(val1 - val2)

        return int(min(self.E_Smooth_threshold, diff))


    def add_aux_node(self, G, aux_name, pred, neigh_pred, alpha, node_index, neigh_index):
        G.add_node(aux_name)
        # node index to aux
        e_smooth = self.calculate_energy_smooth(pred, alpha)
        G.add_edges_from([(node_index, aux_name, {"capacity": e_smooth}),
                        ('aux_name', node_index, {"capacity": e_smooth})])

        # neighbor to aux
        e_smooth = self.calculate_energy_smooth(neigh_pred, alpha)
        G.add_edges_from([(neigh_index, aux_name, {"capacity": e_smooth}),
                        ('aux_name', neigh_index, {"capacity": e_smooth})])

        # aux to not alpha
        e_smooth = self.calculate_energy_smooth(pred, neigh_pred)
        G.add_edges_from([('not_alpha', aux_name, {"capacity": e_smooth}),
                        (aux_name, 'not_alpha', {"capacity": e_smooth})])


    def add_neigh_node(self, G, pred, alpha, node_index, neigh_index):
        e_smooth = self.calculate_energy_smooth(pred, alpha)
        G.add_edge(node_index, neigh_index, capacity=e_smooth)
        G.add_edge(neigh_index, node_index, capacity=e_smooth)

    def alpha_expansion(self, alpha):
        (height, width) = self.shape_original

        # Create graph
        G = nx.DiGraph()
        G.add_nodes_from(list(range(self.image_length)) + ['alpha', 'not_alpha'])

        # Start to optimize
        for node_index, pred in enumerate(self.preds):
            # print()
            # Use np.unravel_index to get the original indices
            (row, col) = np.unravel_index(node_index, shape=self.shape_original)
            col_shifted = col + self.ndisp

            # Create t-links connected to the terminals αlpha and not_alpha
            # Caldulate the data energy
            patch1 = self.left_image[row:row+self.kernel, col_shifted:col_shifted+self.kernel]
            patch2 = self.right_image[row:row+self.kernel, col_shifted-alpha:col_shifted-alpha+self.kernel]

            data_energy = self.calculate_energy_data(patch1, patch2)

            # Adding edges for 'alpha'
            G.add_edges_from([(node_index, 'alpha', {"capacity": data_energy}),
                            ('alpha', node_index, {"capacity": data_energy})])

            # Adding edges for 'not alpha'
            # if it's correct prediction, then the edge to not_alpha should be unaccessible
            if pred == alpha:
                data_energy = float('inf')
            else:
                # else: we calculate the data energy and assign it to the t link
                patch2 = self.right_image[row:row+self.kernel, col_shifted-pred:col_shifted+self.kernel-pred]
                data_energy = self.calculate_energy_data(patch1, patch2)
            G.add_edges_from([(node_index, 'not_alpha', {"capacity": data_energy}),
                            ('not_alpha', node_index, {"capacity": data_energy})])

            # Create n-links connected to neighbors, avoid repitation -> bottom and right
            # Check 1: the bottom neighbor
            if height - node_index // width > 1:    # check if inside boundry and it's not the bottom row
                neigh_index = node_index + width
                neigh_pred = self.preds[neigh_index]
                if pred == neigh_pred:
                    self.add_neigh_node(G, pred, alpha, node_index, neigh_index)
                else:
                    aux_name = 'node_{}_{}'.format(node_index, neigh_index)
                    self.add_aux_node(G, aux_name, pred, neigh_pred, alpha, node_index, neigh_index)

            # Check 2: the right neighbor
            neigh_index = node_index + 1
            if neigh_index % width > 0 :          # check if inside boundry and it's not the right edge
                neigh_pred = self.preds[neigh_index]
                if pred == neigh_pred:
                    self.add_neigh_node(G, pred, alpha, node_index, neigh_index)
                else:
                    aux_name = 'node_{}_{}'.format(node_index, neigh_index)
                    self.add_aux_node(G, aux_name, pred, neigh_pred, alpha, node_index, neigh_index)

        initial_alpha_size = np.sum(self.preds == alpha)
        # perform min-cut
        min_cut_val, success = self.perform_min_cut(G, alpha, initial_alpha_size)
        # TODO: Add early stop?

        return success


    def is_success(self, initial_alpha_size, partition_len, min_cut_val):
        # TODO add last min cut
        # print('mincut value', min_cut_val)
        # print('initial correct numbers', initial_alpha_size)
        # print('after optimization', partition_len)
        if initial_alpha_size < partition_len:
            return True
        return False

    def perform_min_cut(self, G, alpha, initial_alpha_size):
        min_cut_val, partitions = nx.minimum_cut(G, 'alpha', 'not_alpha', capacity='capacity')
        # Do not include the non int nodes (aux nodes)
        partition = [x for x in partitions[1] if isinstance(x, int)]
        self.preds[partition] = alpha
        success = self.is_success(initial_alpha_size, len(partition), min_cut_val)
        return min_cut_val, success

    # the object calls this function to perform optimization
    def perform_alpha_expansion(self):
        random.shuffle(self.labels)
        for label in self.labels:
            self.alpha_expansion(label)
        return np.reshape(self.preds, newshape=self.shape_original)
