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

    ax.scatter(X[:, 0], X[:, 1], s=36, color='orange',linewidth=1,alpha=0.25)
    ax.set_xlim(-3,5)
    ax.spines['bottom'].set_color('black') # 设置画框的底部的颜色
    ax.spines['bottom'].set_linewidth(2.2) # 设置画框的底部的粗细
    ax.spines['left'].set_color('black')
    ax.spines['left'].set_linewidth(2.2)
    ax.spines['top'].set_color('black')
    ax.spines['top'].set_linewidth(2.2)
    ax.spines['right'].set_color('black')
    ax.spines['right'].set_linewidth(2.2)
    ax.axis('equal') # 正方形画框
    ax.xaxis.grid(True,color='black',alpha=0.1,linestyle='-.')
    ax.yaxis.grid(True,color='black',alpha=0.1,linestyle='-.')
    w_factor1 = 0.9 / gmm.weights_.max()
    w_factor2 = 0.1 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor1,color='#fc4e2a',fill=False,linewidth=2,linestyle='dotted')
        draw_ellipse(pos, covar, alpha=w * w_factor2,color='#fc4e2a',fill=True)

    plt.title("Extracted Feature") # 设置标题
    plt.axis('square')
    plt.xlim(-1.05,1.05) # 设置x轴的取值范围
 
    from matplotlib.pyplot import MultipleLocator 
    y_major_locator=MultipleLocator(0.5) # 设置y轴的刻度仅仅显示0.5的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    plt.ylim(-1.05,1.05) # 设置y轴的取值范围

if __name__ == '__main__':
    X, y_true = make_blobs(n_samples=400, centers=2,center_box=(-1,1),
cluster_std=0.25, random_state=0)
    X = X[:, ::-1] # flip axes for better plotting

    gmm = GMM(n_components=2).fit(X)
    labels = gmm.predict(X)

    gmm = GMM(n_components=2, random_state=42)
    plot_gmm(gmm, X)

    plt.show()