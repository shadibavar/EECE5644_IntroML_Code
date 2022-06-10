import numpy as np
from scipy.stats import multivariate_normal # MVN not univariate
import matplotlib.pyplot as plt


def generate_data(mu, Sigma, priors, N):

    u = np.random.rand(N)
    n = mu.shape[1]
    C = len(priors) # how MANY classes there are

    thresholds = np.cumsum(priors) # [odds of 1, odds of 1 or 2, odds of 1 or 2 or 3] 
    thresholds = np.insert(thresholds, 0, 0) # For intervals of classes... So if between thresholds[0] and [1] -> class 1
                                                                                #between thresholds[1] and [2] -> class 2
                                                                                #between thresholds[2] and [3] -> class 3
                                                                                #between thresholds[3] and [4] -> class 4
    # Output samples and labels/values
    X = np.zeros([N, n]) # N by n vector
    y = np.zeros(N) # KEEP TRACK OF THIS, N long vector- we want to label each data point

    L = np.array(range(1,C+1)) # Labels 0,1,2,3 like we want
    for l in L:
        # Get randomly sampled indices for each class
        indices = np.argwhere((thresholds[l-1] <= u) & (u <= thresholds[l]))[:, 0]
        Nl = len(indices)  # number of samples allocated to this class
        y[indices] = l-1 * np.ones(Nl)  # set the labels
        X[indices, :] =  multivariate_normal.rvs(mu[l-1], Sigma[l-1], Nl) #generate the data points

    return [X, y]

marker_shapes = 'd+.x'
marker_colors = 'rbgm' 
def visualize_data(dataList, sampleLengths, L, sampleTypes, verbose=True):
    sampleClassCounts = []

    for i in range(len(sampleLengths)):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection='3d')

        data = dataList[i]
        X = data[0]
        labels = data[1]

        Nl = np.array([sum(labels == l) for l in L]) # count how many of each label are in labels
        sampleClassCounts.append(Nl)
        if verbose:
            print("Number of samples of total {} from Class 1: {:d}, Class 2: {:d}, Class 3: {:d}, Class 4: {:d}".format(sampleLengths[i], Nl[0], Nl[1], Nl[2], Nl[3]))
        for l in L:
            ax.scatter(X[labels==l, 0], X[labels==l, 1], X[labels==l, 2], marker=marker_shapes[l], color=marker_colors[l], label="True Class {}".format(l))

        ax.set_ylabel('x_2')
        ax.set_xlabel('x_1')
        ax.set_zlabel('x_3')
        plt.legend()
        plt.title("{} Data, {} points".format(sampleTypes, sampleLengths[i]))    
    
    return sampleClassCounts