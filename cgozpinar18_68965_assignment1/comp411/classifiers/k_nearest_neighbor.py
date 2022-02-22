from builtins import range
from builtins import object

from matplotlib.pyplot import axis
import numpy as np
from past.builtins import xrange


class KNearestNeighbor(object):
    """ a kNN classifier with L1 and L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0, distfn='L2'):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        if num_loops == 0:
            if distfn == 'L2':
                dists = self.compute_L2_distances_no_loops(X)
            else:
                dists = self.compute_L1_distances_no_loops(X)
        elif num_loops == 1:
            if distfn == 'L2':
                dists = self.compute_L2_distances_one_loop(X)
            else:
                dists = self.compute_L1_distances_one_loop(X)
        elif num_loops == 2:
            if distfn == 'L2':
                dists = self.compute_L2_distances_two_loops(X)
            else:
                dists = self.compute_L1_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_L2_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension, nor use np.linalg.norm().          #
                #####################################################################
                # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

                # -- attain current row elements
                current_test_element = X[i]
                current_train_element = self.X_train[j]

                # -- compute L2(Euclidian) distance
                feature_wise_diff = current_test_element - current_train_element
                feature_wise_diff_squared = feature_wise_diff ** 2
                l2_norm = np.sqrt(sum(feature_wise_diff_squared))

                # -- add computed distance to the list
                dists[i, j] = l2_norm

                # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_L2_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_L2_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            #######################################################################
            # TODO:                                                               #
            # Compute the l2 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            # Do not use np.linalg.norm().                                        #
            #######################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # --
            current_test_element = X[i]

            feature_wise_diff = self.X_train - current_test_element #make use of numpy's Broadcasting!
            l2_norm = np.sqrt(np.sum(feature_wise_diff ** 2, axis=1))

            dists[i] = l2_norm
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        return dists

    def compute_L2_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_L2_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy,                #
        # nor use np.linalg.norm().                                             #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # -- (x-y)^2 = x^2 + y^2 + 2xy
        x_test_sqr = np.sum(X ** 2, axis=1)
        x_train_sqr = np.sum(self.X_train ** 2, axis=1)

        x_test_sqr = np.expand_dims(x_test_sqr, 0).T
        x_train_sqr = np.expand_dims(x_train_sqr, 0)

        xy = np.dot(X, self.X_train.T)

        l2_norm = np.sqrt(x_test_sqr + x_train_sqr - (2 * xy))
        dists = l2_norm        

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_L1_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # Compute the l1 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension, nor use np.linalg.norm().          #
                #####################################################################
                # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

                # -- Get the current elements
                current_test_element = X[i]
                current_train_element = self.X_train[j]

                # -- Compute the L1(Manhattan) distance between current elements(rows)
                feature_wise_diff = current_test_element - current_train_element
                abs_feature_wise_diff = abs(feature_wise_diff)
                l1_norm = sum(abs_feature_wise_diff)

                # -- save the computed distance in the list
                dists[i, j] = l1_norm

                # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_L1_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_L1_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            #######################################################################
            # TODO:                                                               #
            # Compute the l1 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            # Do not use np.linalg.norm().                                        #
            #######################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # --
            current_test_element = X[i]

            feature_wise_diff = self.X_train - current_test_element
            l1_norm = np.sum(np.abs(feature_wise_diff) ,axis=1)

            dists[i] = l1_norm
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_L1_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_L1_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Compute the l1 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy,                #
        # nor use np.linalg.norm().                                             #
        #                                                                       #
        # HINT: Try to formulate the l1 distance using broadcast operations     #
        #                                                                       #
        #########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # --
        # Implementation below requires too much RAM therefore, It is commented out !
        #l1_norm = X[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]
        #l1_norm = np.abs(np.sum(l1_norm, axis=2))
        
        #dists = l1_norm
        

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []
            #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            
            # -- find k nearest neighbors
            sorted_indices = np.argsort(dists[i], axis=0) #sort the row in ascending order
            sorted_k_indices = sorted_indices[:k]
            
            # -- find labels of the k neares neighbors
            closest_y = self.y_train[sorted_k_indices]


            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # -- find the most common label(create a hashmap with key: label, value: frequency) to find the highes frequency label
            label_freq_dict = dict()
            for e in closest_y:
                if e not in label_freq_dict:
                    label_freq_dict[e] = 1
                else:
                    label_freq_dict[e] += 1

            max_k = None
            max_v = -float("inf")
            for key, v in label_freq_dict.items():
                if max_v < v:
                    max_k = key
                    max_v = v

            pred_label = max_k

            # -- store the label
            y_pred[i] = pred_label

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred
