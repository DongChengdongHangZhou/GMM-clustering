import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from sklearn.datasets._samples_generator import make_blobs
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture as GMM
from matplotlib.patches import Ellipse

def draw_ellipse(position, covariance, ax=None, **kwargs):
# """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    ax.set_facecolor((1.0, 1, 1))
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
        # Draw the ellipse
    for nsig in range(1, 3):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,angle, **kwargs))

def plot_gmm(gmm, X):
    ax = plt.gca()
    ax.set_facecolor((1.0, 1, 1))
    labels = gmm.fit(X).predict(X)

    ax.scatter(X[:, 0], X[:, 1], s=6, color='r',alpha=0.2)
    ax.set_xlim(-3,5)
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.axis('equal')
    ax.xaxis.grid(True,color='black',alpha=0.2)
    ax.yaxis.grid(True,color='black',alpha=0.2)
    w_factor1 = 0.7 / gmm.weights_.max()
    w_factor2 = 0.15 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor1,color='green',fill=False)
        draw_ellipse(pos, covar, alpha=w * w_factor2,color='green',fill=True)

if __name__ == '__main__':
    X, y_true = make_blobs(n_samples=400, centers=2,
cluster_std=0.60, random_state=0)
    X = X[:, ::-1] # flip axes for better plotting

    gmm = GMM(n_components=2).fit(X)
    labels = gmm.predict(X)
    # plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
    gmm = GMM(n_components=2, random_state=42)
    plot_gmm(gmm, X)

    plt.show()