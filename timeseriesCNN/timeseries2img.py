'''
Functions for timeseries 2 img
'''
import numpy as np
from pyts.image import GramianAngularField, MarkovTransitionField


class Data: 
    def __init__(self, X, y, image_size):
        '''
        X.shape = (n_samples, n_timestamps) or (n_samples, n_timestamps, n_features)
        y.shape = (n_samples, )
        '''
        self.X = X
        self.y = y

        if type(image_size) == float: # if image_size is a float, we calculate
            self.image_size = int(X.shape[1] * image_size)
        else:
            self.image_size = image_size

        # check if X and y have the same number of samples
        if X.shape[0] != len(y): 
            raise ValueError('X and y should have the same number of samples')

    def scale(): 
        pass

    def convert_to_img(self, methods=["GASF, GADF, MTF"]): 
        """
        """

        # prepare empty numpy array for the images
        n_samples = self.X.shape[0]
        n_methods = len(methods)
        n_features = self.X.shape[2] if len(self.X.shape) == 3 else 1

        im = np.zeros((n_samples, self.image_size, self.image_size, n_features, n_methods)) # img has to be symmetric

        for method_index, method in enumerate(methods):

            if type(method) == str:
                if method == "GASF":
                    transformer = GramianAngularField(image_size=self.image_size, method='summation')
                elif method == "GADF":
                    transformer = GramianAngularField(image_size=self.image_size, method='difference')
                elif method == "MTF":
                    transformer = MarkovTransitionField(image_size=self.image_size)
            else:
                transformer = method

            for feature in range(n_features):
                
                X_tmp = self.X[:,:,feature] if n_features > 1 else self.X

                im[:, :, :, feature, method_index] = transformer.fit_transform(X_tmp)

        return im

if __name__ == "__main__":
    from pyts.datasets import load_gunpoint
    X, _, y, _ = load_gunpoint(return_X_y=True)   

    # make a 3d numpy array for the X
    X = np.random.rand(50, 150, 3)
    print(X.shape, y.shape, len(y))

    # data
    data = Data(X, y, image_size=0.5)

    gaf = data.convert_to_img(methods=["GASF", "GADF", "MTF"])

    print(gaf.shape)

