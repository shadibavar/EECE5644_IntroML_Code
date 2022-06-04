# X in R(2) takes values from mix of 4 gaussians
# Each gaussian is the class-conditional pdf of one of four labels

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal # MVN not univariate
from matplotlib.lines import Line2D


priors = [0.2, 0.25, 0.25, 0.3]

mu = np.array([[-0.2, -0.2], 
               [0.1, 0.11], 
               [0.4, 0.43],
               [0.7, 0.71]]) # distribution means

Sigma = np.array([0.19*np.eye(2),
                  0.17*np.eye(2),
                  0.27*np.eye(2),
                  0.29*np.eye(2)]) # distribution covariances

#print(mu, Sigma)
     
# PART A
# 1. Generate 10k samples and keep track of true labels

N = 10000
n = mu.shape[1]
C = len(priors) # how MANY classes there are
# Decide randomly which samples will come from each component

np.random.seed(9)
u = np.random.rand(N) # N random numbers (0 to 1)
thresholds = np.cumsum(priors) # [odds of 1, odds of 1 or 2, odds of 1 or 2 or 3] 
thresholds = np.insert(thresholds, 0, 0) # For intervals of classes... So if between thresholds[0] and [1] -> class 1
                                                                            #between thresholds[1] and [2] -> class 2
                                                                            #between thresholds[2] and [3] -> class 3
                                                                            #between thresholds[3] and [4] -> class 4

# Output samples and labels
X = np.zeros([N, n]) # N by n vector
labels = np.zeros(N) # KEEP TRACK OF THIS, N long vector- we want to label each data point

# Plot for original data and their true labels
fig = plt.figure(figsize=(10, 10))
marker_shapes = 'd+.x'
marker_colors = 'rbgm' 

L = np.array(range(1, C+1))
for l in L:
    # Get randomly sampled indices for this component
    indices = np.argwhere((thresholds[l-1] <= u) & (u <= thresholds[l]))[:, 0]
        # "if u is between thresholds 0 and 1, 1 and 2, 2 and 3, 3 and 4" (one check per entry in L which is 1 to 4)
        # finds the *indices* where the expression evaluates to TRUE - that is, grabs the indices where u values fell in the range attributed to each class probability 
    # No. of samples in this component
    Nl = len(indices)  
    labels[indices] = l * np.ones(Nl) # since our labels are [1,2,3,4], we can populate the labels vector with these numbers according to l
    # at each index where u is label l (that is, first 1, then 2, then 3 etc), populate the *value* with a number from a gaussian distribution of
    # mean and variance corresponding to that label. 
    X[indices, :] =  multivariate_normal.rvs(mu[l-1], Sigma[l-1], Nl)
    plt.plot(X[labels==l, 0], X[labels==l, 1], marker_shapes[l-1] + marker_colors[l-1], label="True Class {}".format(l))

Nl = np.array([sum(labels == l) for l in L]) # count how many of each label are in labels
print("Number of samples from Class 1: {:d}, Class 2: {:d}, Class 3: {:d}, Class 4: {:d}".format(Nl[0], Nl[1], Nl[2], Nl[3]))

# Plot the original data and their true labels
plt.legend()
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.title("Generated Original Data Samples")
plt.tight_layout()

# 2. Min probability of error is achieved with 0-1 loss, which makes the loss matrix:

# Lamba (4x4) with 0 on diagonals and 1 elsewhere
Lambda_MAP = np.ones([4,4])
np.fill_diagonal(Lambda_MAP,0)
print(Lambda_MAP)

# and the matrix form of the loss minimization decision rule is
# Risk across all decisions = loss matrix * (class likelihoods * class priors) = loss * pirors 

# [[ R(D(x) = 1 | x) ],            [[P(Y = 1 | x) 0 0 0 ],    [[p(x | Y = 1)],
#  [ R(D(x) = 2 | x) ],             [0 P(Y = 2 | x) 0 0 ],     [p(x | Y = 2)],
#  [ R(D(x) = 3 | x) ], = Lambda *  [0 0 P(Y = 3 | x) 0 ],  *  [p(x | Y = 3)], 
#  [ R(D(x) = 4 | x) ]]             [0 0 0 P(Y = 4 | x) ]]     [p(x | Y = 4)]]

Y = np.array(range(C))

# Calculate class-conditional likelihoods p(x|Y=j) for each label of the N observations
class_cond_likelihoods = np.array([multivariate_normal.pdf(X, mu[j], Sigma[j]) for j in Y])
class_priors = np.diag(priors)
print(class_cond_likelihoods.shape)
print(class_priors.shape)
class_posteriors = class_priors.dot(class_cond_likelihoods)
print(class_posteriors) # size (4,10000)

# We want to create the risk matrix of size 4 x N 
cond_risk = Lambda_MAP.dot(class_posteriors)
print(cond_risk)

# Get the decision for each column in risk_mat
# we pick the row with the lowest value, which represents the decision with the lowest risk
decisions = np.argmin(cond_risk, axis=0)
print(decisions.shape)

# Plot for decisions vs true labels
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(projection='3d')

marker_shapes = '.o^s' # Accomodates up to C=5

# Get sample class counts
sample_class_counts = np.array([sum(labels == j+1) for j in Y]) # need to offset to account for label being 1 to 4

# Confusion matrix
conf_mat = np.zeros((C, C))

# 3. Visualize the data.
cumulative_loss = 0
for i in Y: # Each decision option
    for j in Y: # Each class label
        ind_ij = np.argwhere((decisions==i) & (labels==j+1))
        cumulative_loss += Lambda_MAP[i,j]*len(ind_ij)
        conf_mat[i, j] = round(len(ind_ij)/sample_class_counts[j],2) # Average over class sample count

        # True label = Marker shape; Decision = Marker Color
        if i == j:
            marker_colors = 'g'
        else: 
            marker_colors = 'r'

        marker = marker_shapes[j] + marker_colors
        # Plot for original data and their true labels
        ax.scatter(X[ind_ij, 0], X[ind_ij, 1], j, color=marker_colors, marker=marker_shapes[j])
            
ax.set_ylabel('x_2')
ax.set_xlabel('x_1')
ax.set_zlabel('true classes')
ax.set_zticklabels([])


print("Confusion matrix:")
print(conf_mat)

print("Average Loss MAP")
print(cumulative_loss/N)

print("Minimum Probability of Error:")
prob_error = 1 - np.diag(conf_mat).dot(sample_class_counts / N)
print(prob_error)

plt.title("Minimum Probability of Error Classified Sampled Data:  {:.3f}".format(prob_error))
legend_elements = [Line2D([0], [0], marker='.', color='w', label='True Class 1',
                          markerfacecolor='k', markersize=10),
                   Line2D([0], [0], marker='o', color='w', label='True Class 2',
                          markerfacecolor='k', markersize=10),
                    Line2D([0], [0], marker='^', color='w', label='True Class 3',
                          markerfacecolor='k', markersize=10),
                    Line2D([0], [0], marker='s', color='w', label='True Class 4',
                          markerfacecolor='k', markersize=10),
                    Line2D([0], [0], color='g', label='Correct', lw=3),
                    Line2D([0], [0], color='r', label='Incorrect', lw=3)]
plt.legend(handles=legend_elements)
# PART B

# Repeat the above, but now:

Lambda = np.array([ [0, 1, 2, 3],
                    [1, 0, 1, 2],
                    [2, 1, 0, 1],
                    [3, 2, 1, 0]])

# so we are penalizing "bigger" errors more (in the sense of confusing further apart categories)
# (i, j)th entry is CLASS i and LABEL j

# Create the new risk matrix of size 4 x N 
cond_risk = Lambda.dot(class_posteriors)
print(cond_risk)

# Get the decision for each column in risk_mat
# we pick the row with the lowest value, which represents the decision with the lowest risk
decisions = np.argmin(cond_risk, axis=0)
print(decisions.shape)

# Plot for decisions vs true labels
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(projection='3d')

marker_shapes = '.o^s' # Accomodates up to C=5
#marker_colors = 'brgmy'

# Get sample class counts
sample_class_counts = np.array([sum(labels == j+1) for j in Y]) # need to offset to account for label being 1 to 4

# Confusion matrix
conf_mat = np.zeros((C, C))

# 3. Visualize the data.
cumulative_loss = 0

for i in Y: # Each decision option
    for j in Y: # Each class label
        ind_ij = np.argwhere((decisions==i) & (labels==j+1))
        cumulative_loss += Lambda[i,j]*len(ind_ij)
        conf_mat[i, j] = round(len(ind_ij)/sample_class_counts[j],2) # Average over class sample count

        # True label = Marker shape; Decision = Marker Color
        if i == j:
            marker_colors = 'g'
        else: 
            marker_colors = 'r'

        marker = marker_shapes[j] + marker_colors
        #plt.plot(X[ind_ij, 0], X[ind_ij, 1], marker)
        ax.scatter(X[ind_ij, 0], X[ind_ij, 1], j, color=marker_colors, marker=marker_shapes[j])

ax.set_zticklabels([])
ax.set_ylabel('x_2')
ax.set_xlabel('x_1')
ax.set_zlabel('true classes')

legend_elements = [Line2D([0], [0], marker='.', color='w', label='True Class 1',
                          markerfacecolor='k', markersize=10),
                   Line2D([0], [0], marker='o', color='w', label='True Class 2',
                          markerfacecolor='k', markersize=10),
                    Line2D([0], [0], marker='^', color='w', label='True Class 3',
                          markerfacecolor='k', markersize=10),
                    Line2D([0], [0], marker='s', color='w', label='True Class 4',
                          markerfacecolor='k', markersize=10),
                    Line2D([0], [0], color='g', label='Correct', lw=3),
                    Line2D([0], [0], color='r', label='Incorrect', lw=3)]

plt.legend(handles=legend_elements)

print("Confusion matrix:")
print(conf_mat)

print("Average loss:")
avg_loss = cumulative_loss/N
print(avg_loss)

print("Minimum Probability of Error:")
prob_error = 1 - np.diag(conf_mat).dot(sample_class_counts / N)
print(prob_error)

plt.title("Minimum Probability of Error Classified Sampled Data:  {:.3f}".format(prob_error))
plt.show()

# noticing that there are not that many points that have been misclassified between 1&3, 1&3, 1&4, 4&1, 2&4, 4&2
# that is, the entries representing the overlap between "distant" classes 
# thus the error estimate will not be that different, since the overall loss will not be that much higher 
# we can increase the significance of the difference between the cases by increasing the amount of overlap between classes