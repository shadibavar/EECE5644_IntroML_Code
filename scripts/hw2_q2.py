import matplotlib.pyplot as plt
import numpy as np


def hw2q2():
    Ntrain = 100
    data = generateData(Ntrain)
    plot3(data[:, 0], data[:, 1], data[:, 2], name="Training")
    xTrain = data[:, 0:2]
    yTrain = data[:, 2]

    Ntrain = 1000
    data = generateData(Ntrain)
    plot3(data[:, 0], data[:, 1], data[:, 2], name="Validation")
    xValidate = data[:, 0:2]
    yValidate = data[:, 2]

    return xTrain, yTrain, xValidate, yValidate


def generateData(N):
    gmmParameters = {}
    gmmParameters['priors'] = [.3, .4, .3]  # priors should be a row vector
    gmmParameters['meanVectors'] = np.array([[-10, 0, 10], [0, 0, 0], [10, 0, -10]])
    gmmParameters['covMatrices'] = np.zeros((3, 3, 3))
    gmmParameters['covMatrices'][:, :, 0] = np.array([[1, 0, -3], [0, 1, 0], [-3, 0, 15]])
    gmmParameters['covMatrices'][:, :, 1] = np.array([[8, 0, 0], [0, .5, 0], [0, 0, .5]])
    gmmParameters['covMatrices'][:, :, 2] = np.array([[1, 0, -3], [0, 1, 0], [-3, 0, 15]])
    X = generateDataFromGMM(N, gmmParameters)
    return X


def generateDataFromGMM(N, gmmParameters):
    #    Generates N vector samples from the specified mixture of Gaussians
    #    Returns samples and their component labels
    #    Data dimensionality is determined by the size of mu/Sigma parameters
    priors = gmmParameters['priors']  # priors should be a row vector
    meanVectors = gmmParameters['meanVectors']
    covMatrices = gmmParameters['covMatrices']
    n = meanVectors.shape[0]  # Data dimensionality
    C = len(priors)  # Number of components
    X = np.zeros((n, N))
    labels = np.zeros((1, N))
    # Decide randomly which samples will come from each component
    u = np.random.random((1, N))
    thresholds = np.zeros((1, C + 1))
    thresholds[:, 0:C] = np.cumsum(priors)
    thresholds[:, C] = 1
    for l in range(C):
        indl = np.where(u <= float(thresholds[:, l]))
        Nl = len(indl[1])
        labels[indl] = (l + 1) * 1
        u[indl] = 1.1
        X[:, indl[1]] = np.transpose(np.random.multivariate_normal(meanVectors[:, l], covMatrices[:, :, l], Nl))

    # NOTE TRANPOSE TO GO TO SHAPE (N, n)
    return X.transpose()


def plot3(a, b, c, name="Training", mark="o", col="b"):
    # Adjusts the aspect ratio and enlarges the figure (text does not enlarge)
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(a, b, c, marker=mark, color=col)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_zlabel(r"$y$")
    plt.title("{} Dataset".format(name))
    # To set the axes equal for a 3D plot
    ax.set_box_aspect((np.ptp(a), np.ptp(b), np.ptp(c)))
    #plt.show()

def plot3_combine(a, b, c, d, e, f, name=["Training","Validation"], mark="oo", col="bc"):
    # Adjusts the aspect ratio and enlarges the figure (text does not enlarge)
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(a, b, c, marker=mark[0], color=col[0], label=name[0])
    ax.scatter(d, e, f, marker=mark[1], color=col[1], label=name[1])

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_zlabel(r"$y$")
    plt.title("{} and {} Dataset".format(name[0], name[1]))
    # To set the axes equal for a 3D plot
    ax.set_box_aspect((np.ptp(a), np.ptp(b), np.ptp(c)))
    plt.legend()
    #plt.show()


def analytical_solution_ML(X, y):
    # Analytical solution is (X^T*X)^-1 * X^T * y 
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

def analytical_solution_MAP(X, y, l):
    # Analytical solution is (X^T*X + lambda*I)^-1 * X^T * y
    return np.linalg.inv(X.T.dot(X)+ l*np.eye(X.shape[1])).dot(X.T).dot(y)


if __name__ == '__main__':
    xTrain, yTrain, xValidate, yValidate = hw2q2()
    Ntrain = xTrain.shape[0]
    Nvalid = xValidate.shape[0]

    x1 = xTrain[:,0]
    x2 = xTrain[:,1]
    zeroth = np.ones((Ntrain,))
    XTrain = np.array([zeroth, x1**3, x2**3, x1**2*x2, x1*x2**2, x1**2, x2**2, x1*x2, x1, x2]).T #set this up as a linear comb matrix
    theta_opt = analytical_solution_ML(XTrain, yTrain)
    
    x1 = xValidate[:,0]
    x2 = xValidate[:,1]
    zeroth = np.ones((Nvalid,))
    XValid = np.array([zeroth, x1**3, x2**3, x1**2*x2, x1*x2**2, x1**2, x2**2, x1*x2, x1, x2]).T #set this up as a linear comb matrix

    ML_preds = XValid.dot(theta_opt)
    MSE_ML = np.mean((yValidate-ML_preds)**2)
    plot3_combine(x1, x2, ML_preds, xValidate[:,0], xValidate[:,1], yValidate, ["Validate","Estimated ML"], mark='..' ,col='bc')

    MSE_MAP = []
    thetas_MAP = []
    gammas = np.logspace(-4, 4, 1000)
    for g in gammas:
        lamba = 1/g
        theta_opt = analytical_solution_MAP(XTrain, yTrain, lamba)
        MAP_preds = XValid.dot(theta_opt)
        MSE_MAP.append(np.mean((yValidate-MAP_preds)**2))
        thetas_MAP.append(theta_opt)
    
    min_MSE_ix = int(np.argwhere(MSE_MAP == min(MSE_MAP)))
    MAP_preds = XValid.dot(thetas_MAP[min_MSE_ix])
    plot3_combine(x1, x2, MAP_preds, xValidate[:,0], xValidate[:,1], yValidate, ["Validate","Estimated MAP"], mark='..' ,col='bc')

    fig = plt.figure()
    plt.plot((gammas), MSE_MAP, label='MAP')
    plt.plot((gammas), MSE_ML*np.ones((len(gammas,))), label='ML')
    plt.xscale('log')
    plt.xlabel("gamma values")
    plt.ylabel("MSE")
    plt.title("MSE as function of gamma for MAP estimation")
    plt.legend()

    print('MSE for Max Likelihood Parameter Estimator: ', MSE_ML)
    print('min MSE for Max A Posteriori Parameter Estimator: ', MSE_MAP[min_MSE_ix], 'at gamma, ', gammas[min_MSE_ix])

    # MAP continuously improves as gamma increases. 
    # gamma is strength of the regularizer. So wer're making it more regularized and this is reducing error? Why?
    # ML is the same as MAP with gamma = 0 and this can be seen in the plot.
    # ALso from the analytical equations.

    plt.show()
    

