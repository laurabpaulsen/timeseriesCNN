'''
Functions for timeseries 2 img
'''
import numpy as np
from pyts.image import GramianAngularField, MarkovTransitionField

from utils import _check_valid_types

class Data: 
    def __init__(self, X:np.array, y:np.array):
        '''
        X.shape = (n_samples, n_timestamps) or (n_samples, n_timestamps, n_features)
        y.shape = (n_samples, )
        '''
        self.X = X
        self.y = y
        self.X_img = None

        # check if X and y have the same number of samples
        if X.shape[0] != len(y): 
            raise ValueError('X and y should have the same number of samples')

    def transform_to_img(self, methods:list|str=["GASF, GADF, MTF"], image_size:float|int=1.0): 
        '''
        Convert the timeseries to images using the methods specified in the methods list.
        '''
        # if methods is a string, convert it to a list
        if type(methods) == str:
            methods = [methods]

        # check if the methods are valid
        _check_valid_types(methods, ["GASF", "GADF", "MTF"])

        # prepare image size (convert to int if it is a float)
        if type(image_size) == float: # if image_size is a float, we calculate the image size
            image_size = int(X.shape[1] * image_size)

        # prepare empty numpy array for the images
        n_samples = self.X.shape[0]
        n_methods = len(methods)
        n_features = self.X.shape[2] if len(self.X.shape) == 3 else 1    

        im = np.zeros((n_samples, image_size, image_size, n_features, n_methods)) # img has to be symmetric

        for method_index, method in enumerate(methods): ## loop over all the methods

            # make more elegant later!
            if type(method) == str:
                if method == "GASF":
                    transformer = GramianAngularField(image_size=image_size, method='summation')
                elif method == "GADF":
                    transformer = GramianAngularField(image_size=image_size, method='difference')
                elif method == "MTF":
                    transformer = MarkovTransitionField(image_size=image_size)
            else: # allows users to pass the transformers directly to include custom keyword arguments
                transformer = method

            for feature in range(n_features):
                
                X_tmp = self.X[:,:,feature] if n_features > 1 else self.X

                im[:, :, :, feature, method_index] = transformer.fit_transform(X_tmp)

        self.X_img = im

if __name__ == "__main__":

    # for testing (keep for now, but remove later)
    from pyts.datasets import load_gunpoint
    X, _, y, _ = load_gunpoint(return_X_y=True)   

    # make a 3d numpy array for the X
    X = np.random.rand(50, 150, 3)
    print(X.shape, y.shape, len(y))

    # data
    data = Data(X, y)

    data.transform_to_img(methods=["GASF", "GADF", "MTF"], image_size=0.5)

    print(data.X_img.shape)

