import numpy as np
import matplotlib.pyplot as plt

"""
Implementation of Principal Component Analysis.
"""
class PCA:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.mean = None
        self.components = None

    def fit(self, X: np.ndarray) -> None:
        #TODO: 10%
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        matrix = np.dot(X_centered.T, X_centered)
        #print(matrix.shape[0], matrix.shape[1])
        
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        idx = np.argsort(eigenvalues)[::-1]
        sorted_eigenvectors = eigenvectors[:, idx]
        
        self.components = sorted_eigenvectors[:, :self.n_components]
        
        '''
        #plot the mean vector as a image, title = Mean Vector Image
        plt.imshow(self.mean.reshape(61, 80),cmap='gray')
        plt.title('Mean Vector Image')
        plt.show()
        
        #plot the first 4 eigenvectors as images, title = Eigenvector Image i (i=1,2,3,4)
        for i in range(4):
            plt.imshow(self.components[:,i].reshape(61, 80),cmap='gray')
            plt.title('Eigenvector Image '+str(i+1))
            plt.show()
        '''
        
        #print(self.components.shape[0], self.components.shape[1])
        #raise NotImplementedError

    def transform(self, X: np.ndarray) -> np.ndarray:
        #TODO: 2%
        X = X - self.mean
        trans = np.dot(self.components.T, X.T).T
        return trans
        #raise NotImplementedError

    def reconstruct(self, X):
        return np.dot(self.components ,self.transform(X)) + self.mean
        #raise NotImplementedError
        #TODO: 2%
