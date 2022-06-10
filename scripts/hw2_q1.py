import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal # MVN not univariate
from matplotlib.lines import Line2D
from sys import float_info # Threshold smallest positive floating value
from math import ceil, floor 

# Generate data and keep track of true labels
np.random.seed(7)

# Wrap this specific distribution into function to make all the data sets:
def generate_data(N):
    priors = [0.65, 0.35]
    weights = [0.5, 0.5]
    mu = np.array([[3, 0], 
                [0, 3], 
                [2, 2]]) # distribution means

    Sigma = np.array([[[2, 0],[0, 1]],
                    [[1, 0],[0, 2]],
                    [[1, 0],[0, 1]]]) # distribution covariances

    n = mu.shape[1]
    C = len(priors) # how MANY classes there are
    # Decide randomly which samples will come from each component

    u = np.random.rand(N) # N random numbers (0 to 1)

    thresholds = np.cumsum([priors[0]*weights[0], priors[0]*weights[1], priors[1]]) 
    thresholds = np.insert(thresholds, 0, 0) # For intervals of classes... So if between thresholds[0] and [1] -> class 0, gaussian 0
                                                                                #between thresholds[1] and [2] -> class 0, gaussian 1
                                                                                #between thresholds[2] and [3] -> class 1
    # Output samples and labels
    X = np.zeros([N, n]) # N by n vector
    labels = np.zeros(N) # KEEP TRACK OF THIS, N long vector- we want to label each data point
    L = np.array(range(1, C+2))
    # print(L)
    classes = [0, 0, 1] # use to set labels correctly, related to thresholds 

    for l in L:
        # Get randomly sampled indices for this component
        indices = np.argwhere((thresholds[l-1] <= u) & (u <= thresholds[l]))[:, 0]
            # "if u is between thresholds 0 and 1, 1 and 2, 2 and 3, 3 and 4" (one check per entry in L which is 1 to 4)
            # finds the *indices* where the expression evaluates to TRUE - that is, grabs the indices where u values fell in the range attributed to each class probability 
        # No. of samples in this component
        Nl = len(indices)  
        # print(Nl)
        labels[indices] = classes[l-1] * np.ones(Nl) 
        X[indices, :] =  multivariate_normal.rvs(mu[l-1], Sigma[l-1], Nl)

    if N==0:
        return priors, weights, mu, Sigma
    else:
        return X, labels

D_20, labels_20 = generate_data(20)
D_200, labels_200 = generate_data(200)
D_2k, labels_2k = generate_data(2000)
D_v, labels_v = generate_data(10000)

priors, weights, mu, Sigma = generate_data(0)
C = 2
L = np.array(range(C))
N = 10000
n = mu.shape[1]

# PART 1
# The classifier is the optimal bayes classifier, which determines MAP,
# and is equivalent to minimizing probability of error for 0-1 loss matrix.
# This can be expressed as: (see notes)

# get the class conditional likelihoods
class_cond_likelihood_0 = weights[0]*multivariate_normal.pdf(D_v, mu[0], Sigma[0]) + weights[1]*multivariate_normal.pdf(D_v, mu[1], Sigma[1])
class_cond_likelihood_1 = multivariate_normal.pdf(D_v, mu[2], Sigma[2])
class_cond_likelihoods = np.array([class_cond_likelihood_0, class_cond_likelihood_1])

# calculate the posteriors
class_priors = np.diag(priors)
# print(class_cond_likelihoods.shape)
# print(class_priors.shape)
class_posteriors = class_priors.dot(class_cond_likelihoods)

# zero-one loss for min error
Lambda_MAP = np.ones((2,2))-np.eye(2)
# reate the risk matrix
cond_risk = Lambda_MAP.dot(class_posteriors)

# Get the decision for each column in risk_mat
# we pick the row with the lowest value, which represents the decision with the lowest risk
decisions = np.argmin(cond_risk, axis=0)
# TODO multiply by priors here???
discriminant_score = (class_cond_likelihoods[1])/(class_cond_likelihoods[0])
# print('decisions shape', decisions.shape)

marker_shapes = '.o^s' # Accomodates up to C=5

# Get sample class counts
sample_class_counts = np.array([sum(labels_v == j) for j in L]) # need to offset to account for label being 1 to 4
print('sample counts:', sample_class_counts)
# Confusion matrix
conf_mat = np.zeros((C, C))

# 3. Visualize the data.
cumulative_loss = 0
# Plot for decisions vs true labels
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(projection='3d')
for i in L: # Each decision option
    for j in L: # Each class label
        ind_ij = np.argwhere((decisions==i) & (labels_v==j))
        cumulative_loss += Lambda_MAP[i,j]*len(ind_ij)
        conf_mat[i, j] = round(len(ind_ij)/sample_class_counts[j],5) # Average over class sample count

        # True label = Marker shape; Decision = Marker Color
        if i == j:
            marker_colors = 'g'
        else: 
            marker_colors = 'r'

        marker = marker_shapes[j] + marker_colors
        # Plot for original data and their true labels
        ax.scatter(D_v[ind_ij, 0], D_v[ind_ij, 1], j, color=marker_colors, marker=marker_shapes[j])
            
ax.set_ylabel('x_2')
ax.set_xlabel('x_1')
ax.set_zlabel('true classes')
ax.set_zticklabels([])

print("Confusion matrix:")
print(conf_mat)

# print("Average Loss MAP")
# print(cumulative_loss/N)

print("Minimum Probability of Error:")
prob_error = 1 - np.diag(conf_mat).dot(sample_class_counts / N)
print(prob_error)

# Plot for original data and their true labels
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot()
# ax.scatter(D_20[labels_20==0, 0], D_20[labels_20==0, 1], c='b', marker='.', label="Class 0")
# ax.scatter(D_20[labels_20==1, 0], D_20[labels_20==1, 1], c='c', marker='+', label="Class 1")

# plt.legend()
# plt.xlabel(r"$x_1$")
# plt.ylabel(r"$x_2$")
# plt.title("Data and True Class Labels")
# plt.tight_layout()

# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot()
# ax.scatter(D_200[labels_200==0, 0], D_200[labels_200==0, 1], c='b', marker='.', label="Class 0")
# ax.scatter(D_200[labels_200==1, 0], D_200[labels_200==1, 1], c='c', marker='+', label="Class 1")

# plt.legend()
# plt.xlabel(r"$x_1$")
# plt.ylabel(r"$x_2$")
# plt.title("Data and True Class Labels")
# plt.tight_layout()

# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot()
# ax.scatter(D_2k[labels_2k==0, 0], D_2k[labels_2k==0, 1], c='b', marker='.', label="Class 0")
# ax.scatter(D_2k[labels_2k==1, 0], D_2k[labels_2k==1, 1], c='c', marker='+', label="Class 1")

# plt.legend()
# plt.xlabel(r"$x_1$")
# plt.ylabel(r"$x_2$")
# plt.title("Data and True Class Labels")
# plt.tight_layout()

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot()
ax.scatter(D_v[labels_v==0, 0], D_v[labels_v==0, 1], c='b', marker='.', label="Class 0")
ax.scatter(D_v[labels_v==1, 0], D_v[labels_v==1, 1], c='c', marker='+', label="Class 1")

plt.legend()
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.title("Data and True Class Labels")
plt.tight_layout()

# scores are independent of gamma; labels are known from the data 
def estimate_roc(discriminant_score, label):
    Nlabels = np.array((sum(label == 0), sum(label == 1)))
    sorted_score = sorted(discriminant_score) #sort so they're in order

    # Use tau values that will account for every possible classification split
    taus = ([sorted_score[0] - float_info.epsilon] + 
             sorted_score +
             [sorted_score[-1] + float_info.epsilon])
   
    # Calculate the decision label for each observation for each gamma
    decisions = [discriminant_score >= t for t in taus]

    # false positives:
    ind10 = [np.argwhere((d==1) & (label==0)) for d in decisions]
    p10 = [len(inds)/Nlabels[0] for inds in ind10]
    # true positives
    ind11 = [np.argwhere((d==1) & (label==1)) for d in decisions]
    p11 = [len(inds)/Nlabels[1] for inds in ind11]
    # false negatives
    ind01 = [np.argwhere((d==0) & (label==1)) for d in decisions]
    p01 = [len(inds)/Nlabels[1] for inds in ind01]
    # true negatives
    ind00 = [np.argwhere((d==0) & (label==0)) for d in decisions]
    p00 = [len(inds)/Nlabels[0] for inds in ind00]

    # ROC has FPR on the x-axis and TPR on the y-axis
    roc = np.array((p10, p11, p01, p00))

    return roc, taus

# Construct the ROC for ERM by changing log(gamma)
roc_erm, gamma_range = estimate_roc(discriminant_score, labels_v)

prob_error_erm = (roc_erm[0]*sample_class_counts[0] + roc_erm[2]*sample_class_counts[1])/N
opt_gamma_ix = np.argwhere(prob_error_erm == min(prob_error_erm))[0]
print('opt gamma ix', opt_gamma_ix)
print("Optimal gamma ERM is: {}".format(gamma_range[int(opt_gamma_ix)]))
print("Minimum probabilty of error ERM is: {}".format(prob_error_erm[int(opt_gamma_ix)]))

gamma_theoretical = priors[0]/priors[1]
print("Optimal gamma (theoretical) from MAP: ", gamma_theoretical)
decision_theoretical = discriminant_score >= gamma_theoretical
# prob false positives:
pFP = sum(((decision_theoretical==1) & (labels_v==0)))/sample_class_counts[0]
# prob false negatives
pFN = sum(((decision_theoretical==0) & (labels_v==1)))/sample_class_counts[1]
pTP = sum(((decision_theoretical==1) & (labels_v==1)))/sample_class_counts[1]

prob_error_theoretical = pFP*priors[0] + pFN*priors[1]
print("Minimum probabilty of error (theoretical) from MAP: {}".format(prob_error_theoretical))

# fig_roc, ax_roc = plt.subplots(figsize=(10, 10))
# ax_roc.plot(roc_erm[0], roc_erm[1], 'b')
# ax_roc.set_xlabel(r"Probability of false alarm $P(D=1|L=0)$")
# ax_roc.set_ylabel(r"Probability of correct decision $P(D=1|L=1)$")
# plt.grid(True)

# # plot onto the figure
# ax_roc.plot(roc_erm[0][opt_gamma_ix], roc_erm[1][opt_gamma_ix], 'b.', label="Minimum P(Error) ERM", markersize=16)
# ax_roc.plot(pFP, pTP, 'r.', label="Minimum P(Error) Theoretical", markersize=16)

# ax_roc.legend()

# PART 2

# 1. Using MLE                                 
# 2. approximate class label posteriors        
# 3. through logistic-linear-function approx    
# 4. and optimize by minimizing NLL       
# 1.: start from theta_ml = argmax ln( p( D | theta ))
#     make iid and independence assumptions, sub some stuff, simplify to:
#      theta_ml = argmin -1/N sum( y * ln g(xn; theta) + (1-y)ln(1-g(xn;theta)))
#     which for linear model has g(x, w) = 1/ 1+exp(-w.T*b) = g(x*w)

# Define the logistic/sigmoid function
def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

# Define the prediction function y = 1 / (1 + np.exp(-X*theta))
# X.dot(theta) inputs to the sigmoid referred to as logits
def predict_prob(X, theta):
    logits = X.dot(theta)
    return sigmoid(logits) # equivalent to g(x, w) = 1/ 1+exp(-w.T*b)

def log_reg_loss(theta, X, y):
    # Size of batch
    B = X.shape[0]

    # Logistic regression model g(X * theta)
    predictions = predict_prob(X, theta)

    # NLL loss, 1/N sum [y*log(g(X*theta)) + (1-y)*log(1-g(X*theta))]
    error = predictions - y
    nll = -np.mean(y*np.log(predictions) + (1 - y)*np.log(1 - predictions))
    
    # Partial derivative for GD
    g = (1 / B) * X.T.dot(error)
    
    # Logistic regression loss, NLL (binary cross entropy is another interpretation)
    return nll, g

# X is samples, y is labels. 
# now *do optimization* using log_reg_loss as the loss function

def GD(theta0, X, y, *args, **kwargs):
    # unpack options/set defaults:
    max_epoch = kwargs['max_epoch'] if 'max_epoch' in kwargs else 200
    alpha = kwargs['alpha'] if 'alpha' in kwargs else 0.1
    epsilon = kwargs['tolerance'] if 'tolerance' in kwargs else 1e-6

    #initialize theta
    theta = theta0

    #let's keep track of some stuff
    trace = {}
    trace['loss'] = []
    trace['theta'] = []
    trace['gradient'] = []

    for epoch in range(1, max_epoch + 1):
        # do a bit of scheduling to help things go faster
        if epoch < 1000:
            a = alpha[0]
        elif epoch < 2000:
            a = alpha[1]
        else:
            a = alpha[2]

        #print("epoch is: ", epoch)
        loss_epoch = 0

        loss, gradient = log_reg_loss(theta, X, y)
        loss_epoch += loss #useful if using batches later on
        
        # Steepest descent update
        theta = theta - a * gradient

        if np.linalg.norm(gradient) < epsilon:
            print('reached minimum tolerance')
            break

        # Storing the history of the parameters and loss values per epoch
        trace['loss'].append(np.mean(loss_epoch))
        trace['theta'].append(theta)
        trace['gradient'].append(gradient)

    print('exiting with gradient', gradient)
    return theta, trace

def quadratic_transformation(X):
    n = X.shape[1]
    phi_X = X
    
    # Take all monic polynomials for a quadratic
    phi_X = np.column_stack((phi_X, X[:, 1] * X[:, 1], X[:, 1] * X[:, 2], X[:, 2] * X[:, 2]))
        
    return phi_X

def create_prediction_score_grid(theta, poly_type='L'):
    # Create coordinate matrices determined by the sample space; can add finer intervals than 100 if desired
    xx, yy = np.meshgrid(np.linspace(bounds_X[0], bounds_X[1], 200), np.linspace(bounds_Y[0], bounds_Y[1], 200))

    # Augment grid space with bias ones vector and basis expansion if necessary
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_aug = np.column_stack((np.ones(200*200), grid)) 
    if poly_type == 'Q':
        grid_aug = quadratic_transformation(grid_aug)

    # Z matrix are the predictions resulting from sigmoid on the provided model parameters
    Z = predict_prob(grid_aug, theta).reshape(xx.shape)
    
    return xx, yy, Z

def plot_prediction_contours(X, theta, ax, poly_type='L'):
    xx, yy, Z = create_prediction_score_grid(theta, poly_type)
    # Once reshaped as a grid, plot contour of probabilities per input feature (ignoring bias)
    cs = ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.55)
    ax.set_xlim([bounds_X[0], bounds_X[1]])
    ax.set_ylim([bounds_Y[0], bounds_Y[1]])

def report_logistic_classifier_results(X, theta, labels, N_labels, ax, poly_type='L'):
    """
    Report the probability of error and plot the classified data, plus predicted 
    decision contours of the logistic classifier applied to the data given.
    """
    
    predictions = predict_prob(X, theta)  
    # Predicted decisions based on the default 0.5 threshold (higher probability mass on one side or the other)
    decisions = np.array(predictions >= 0.5)
    
    # True Negative Probability Rate
    ind_00 = np.argwhere((decisions == 0) & (labels == 0))
    tnr = len(ind_00) / N_labels[0]
    # False Positive Probability Rate
    ind_10 = np.argwhere((decisions == 1) & (labels == 0))
    fpr = len(ind_10) / N_labels[0]
    # False Negative Probability Rate
    ind_01 = np.argwhere((decisions == 0) & (labels == 1))
    fnr = len(ind_01) / N_labels[1]
    # True Positive Probability Rate
    ind_11 = np.argwhere((decisions == 1) & (labels == 1))
    tpr = len(ind_11) / N_labels[1]

    prob_error = fpr*priors[0] + fnr*priors[1]

    print("The total error achieved with this classifier is {:.3f}".format(prob_error))
    
    # Plot all decisions (green = correct, red = incorrect)
    ax.plot(X[ind_00, 1], X[ind_00, 2], 'og', label="Class 0 Correct", alpha=.25)
    ax.plot(X[ind_10, 1], X[ind_10, 2], 'or', label="Class 0 Wrong")
    ax.plot(X[ind_01, 1], X[ind_01, 2], '+r', label="Class 1 Wrong")
    ax.plot(X[ind_11, 1], X[ind_11, 2], '+g', label="Class 1 Correct", alpha=.25)

    # Draw the decision boundary based on whether its linear (L) or quadratic (Q)
    plot_prediction_contours(X, theta, ax, poly_type)
    ax.set_aspect('equal')

def plot_decision_boundaries(X, labels, theta, ax, poly_type='L'): 
    # Plots original class labels and decision boundaries
    ax.plot(X[labels==0, 1], X[labels==0, 2], 'o', label="Class 0")
    ax.plot(X[labels==1, 1], X[labels==1, 2], '+', label="Class 1")
    
    xx, yy, Z = create_prediction_score_grid(theta, poly_type)
    # Once reshaped as a grid, plot contour of probabilities per input feature (ignoring bias)
    cs = ax.contour(xx, yy, Z, levels=1, colors='k')

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_aspect('equal')


# Prepend column of ones to create augmented inputs tilde{x}
D_train = []

D_20_t = np.column_stack((np.ones(20), D_20))  
n = D_20_t.shape[1]
print('n is', n)

D_200_t = np.column_stack((np.ones(200), D_200))  
D_2k_t = np.column_stack((np.ones(2000), D_2k))  
D_valid = np.column_stack((np.ones(10000), D_v))  

# put all the data in one list so we can train more efficiently
D_train.append(D_20_t)
D_train.append(D_200_t)
D_train.append(D_2k_t)

labels_train = []
labels_train.append(labels_20)
labels_train.append(labels_200)
labels_train.append(labels_2k)

#initialize parameters
theta0_linear = np.random.randn(n)

# set desired options
opts = {}
opts['max_epoch'] = 4000
opts['alpha'] = [0.3, 0.1, 0.01] 
opts['tolerance'] = 1e-3

# Use the validation set's sample space to bound the grid of inputs
# Work out bounds that span the input feature space (x_1 and x_2)
bounds_X = np.array((floor(np.min(D_valid[:,1])), ceil(np.max(D_valid[:,1]))))
bounds_Y = np.array((floor(np.min(D_valid[:,2])), ceil(np.max(D_valid[:,2]))))

fig_decision, ax_decision = plt.subplots(2, 3, figsize=(9, 12));

print("training the models!")
N_train = [20,200,2000]
for i in [0,1,2]:
    theta_gd, trace = GD(theta0_linear, D_train[i], labels_train[i], **opts)    
    theta_hist = np.asarray(trace['theta'])
    nll_hist = np.array(trace['loss'])
    print('theta for GD {} is: {}'.format(N_train[i], theta_gd))

    plot_decision_boundaries(D_train[i], labels_train[i], theta_gd, ax_decision[0, i])
    ax_decision[0, i].set_title("Decision Boundary for \n Logistic-Linear Model N={}".format(D_train[i].shape[0]), fontsize=10)

    # Linear: use validation data (10k samples) and make decisions in report results routine
    report_logistic_classifier_results(D_valid, theta_gd, labels_v, sample_class_counts, ax_decision[1, i])
    ax_decision[1, i].set_title("Classifier Decisions on Validation Set \n Logistic-Linear Model N={}".format(N_train[i]), fontsize=10)
    print("\r")

# Again use the most sampled subset (validation) to define x-y limits
plt.setp(ax_decision, xlim=bounds_X, ylim=bounds_Y)

# Adjust subplot positions
# #plt.subplots_adjust(left=0.05,
#                     bottom=0.05, 
#                     right=0.6, 
#                     top=0.95, 
#                     wspace=0.1, 
#                     hspace=0.3)

# Super plot the legends
handles, labels = ax_decision[1, 0].get_legend_handles_labels()
fig_decision.legend(handles, labels, loc='upper right')
fig_decision.tight_layout()

## QUADRATIC
opts['max_epoch'] = 4000
opts['alpha'] = [0.2, 0.1, 0.01] 
opts['tolerance'] = 1e-3

fig_decision, ax_decision = plt.subplots(2, 3, figsize=(9, 12));
fig_history, ax_history = plt.subplots(1, 1, figsize=(7, 10));

theta0_quadratic = np.random.randn(n+3)
for i in [0,1,2]:
    D_quad = quadratic_transformation(D_train[i])
    theta_gd, trace = GD(theta0_quadratic, D_quad, labels_train[i], **opts)    
    theta_hist = np.asarray(trace['theta'])
    nll_hist = np.array(trace['loss'])
    grad_hist = np.array(trace['gradient'])

    print('theta for GD 20 is:', theta_gd)
    ax_history.plot(range(len(nll_hist)), nll_hist)

    plot_decision_boundaries(D_quad, labels_train[i], theta_gd, ax_decision[0, i], poly_type='Q')
    ax_decision[0, i].set_title("Decision Boundary for \n Logistic-Quadratic Model N={}".format(D_train[i].shape[0]), fontsize=10)

    # Linear: use validation data (10k samples) and make decisions in report results routine
    D_valid_quad = quadratic_transformation(D_valid)
    report_logistic_classifier_results(D_valid_quad, theta_gd, labels_v, sample_class_counts, ax_decision[1, i], poly_type='Q')
    ax_decision[1, i].set_title("Classifier Decisions on Validation Set \n Logistic-Quadratic Model N={}".format(N_train[i]), fontsize=10)
    print("\r")

# Again use the most sampled subset (validation) to define x-y limits
plt.setp(ax_decision, xlim=bounds_X, ylim=bounds_Y)

# Adjust subplot positions
# plt.subplots_adjust(left=0.05,
#                     bottom=0.05, 
#                     right=0.6, 
#                     top=0.95, 
#                     wspace=0.1, 
#                     hspace=0.3)

# Super plot the legends
handles, labels = ax_decision[0, 1].get_legend_handles_labels()
fig_decision.legend(handles, labels, loc='upper right')
fig_decision.tight_layout()


plt.show()