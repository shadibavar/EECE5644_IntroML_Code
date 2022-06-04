from cProfile import label
from matplotlib.pyplot import cla
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal # MVN not univariate

# Read the CSV into a pandas data frame (df)
dfd = pd.read_csv('../X_train.txt', sep='\s+')
dfl = pd.read_csv('../y_train.txt', sep='\s+')
dfdte = pd.read_csv('../X_test.txt', sep='\s+')
dflte = pd.read_csv('../y_test.txt', sep='\s+')

dff = pd.read_csv('../features.txt', sep='\s+')
dfa = pd.read_csv('../activity_labels.txt', sep='\s+')

# turn the vals into arrays, merge test and train data: 
X = np.append(dfd.values, dfdte.values, axis=0)
labels = np.append(dfl.values, dflte.values, axis=0)
labels = (labels-1).T[0]
feature_names = dff.values[:,1]
label_names = dfa.values[:,1]
print('label_names', label_names)
print('feature name test', feature_names[333])

N = X.shape[0] # num data points
F = X.shape[1] # num features
print('X shape', X.shape)
print('labels shape', labels.shape)

C = 6 # num classes

L = np.array([l for l in range(C)])

# Using all available samples from a class, estimate mean and 
# covariance matrices w/ samples averages, also class priors.
Nl = np.array([sum(labels == l) for l in L]) # how many are in each class
print("Instances of each class:", Nl)
priors = Nl/N
priors = priors.reshape((1,C))[0]
print("Class priors are:", priors)
#print("Class priors are: C0 = {:.3f}, C1 = {:.3f}, C2 = {:.3f}, C3 = {:.3f}, C4 = {:.3f}, C5 = {:.3f}".format(priors))

# 1. find the indices of the samples in a given class: incides = np.argwhere(labels==l)
# 2. take the mean of each column of X at those indices: mean(X[indices], axis=0)
# 3. take the covariance of those rows of data
# optionally modify to avoid computing on 0-size classes
nonzero_ix = L[np.argwhere(Nl!=0)].T[0]
mu = np.zeros((C,F))
cov = [None]*C

for l in L: # for each CLASS
    if not np.any(nonzero_ix==l): # no data in this class
        print('no data rows in class: ', l)
        mu[l,:] = np.zeros((1,F))
        cov[l] = np.ones((F,F))

    else:
        print(X[np.array(np.nonzero(labels==l))[0,:],:].shape)
        mu[l,:] = np.mean(X[np.array(np.nonzero(labels==l))[0,:],:], axis=0)
        cov[l] = np.cov(X[np.array(np.nonzero(labels==l))[0,:],:], rowvar=False)
        print('data found in class: ', l)

#print(mu)
#print(cov)

conditions = np.array([np.linalg.cond(covm) for covm in cov])
print("Condition nums are: {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}".format(*conditions))

epsilon = 0.1
#[print('lambda is ', epsilon*np.trace(covm)*np.linalg.matrix_rank(covm)) for covm in cov]
#cov_reg = np.array([covm + epsilon*np.trace(covm)/np.linalg.matrix_rank(covm)*np.eye(F,F) for covm in cov])
cov_reg = np.array([covm + epsilon*np.eye(F,F) for covm in cov])

for j in range(C):
    w, v = np.linalg.eig(cov_reg[j]) 
    if any(w)<epsilon:
        print(j, ' not regularized')
    else:
        print(j, ' regularized')

print("New Condition nums: {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}".format(*np.linalg.cond(cov_reg)))

# Implement MPE classifiers
#    - assume each feature distribution in each class is Gaussian
#    - apply MPE classification rule on all samples
#    - count the errors
#    - report the error probability estimate 
#    - report the confusion matrix 
#    - visualize in 2 or 3d projections w/ subsets of features
#        - randomly pick a few combos
#    - perform PCA to do the same plot 

# loss will be 0-1 so we get minimum error possible
Lambda = np.ones((C,C)) - np.eye(C)

# Risk across all decisions = loss matrix * (class likelihoods * class priors) = loss * pirors 

# [[ R(D(x) = 1 | x) ],            [[P(Y = 1 | x) 0 0 0 ],    [[p(x | Y = 1)],
#  [ R(D(x) = 2 | x) ],             [0 P(Y = 2 | x) 0 0 ],     [p(x | Y = 2)],
#  [ R(D(x) = 3 | x) ], = Lambda *  [0 0 P(Y = 3 | x) 0 ],  *  [p(x | Y = 3)], 
#  [ R(D(x) = 4 | x) ]]             [0 0 0 P(Y = 4 | x) ]]     [p(x | Y = 4)]]

# Calculate class-conditional likelihoods p(x|Y=j) for each label of the N observations
#Y = range(len(nonzero_ix))
class_cond_likelihoods = np.array([multivariate_normal.pdf(X, mu[j], cov_reg[j]) for j in L])
class_priors = np.diag(priors)
#print(class_cond_likelihoods.shape)
#print(class_priors.shape)
class_posteriors = class_priors.dot(class_cond_likelihoods)
print("class posteriors: ", class_posteriors) 

# We want to create the risk matrix of size 4 x N 
cond_risk = Lambda.dot(class_posteriors)
print("conditional risk: ", cond_risk)

# Get the decision for each column in risk_mat
# we pick the row with the lowest value, which represents the decision with the lowest risk
decisions = np.argmin(cond_risk, axis=0) # CHECK DIRECTIONS
print("decisions shape: ", decisions.shape)

# Get sample class counts
sample_class_counts = np.array([sum(labels == j) for j in L]) # need to offset to account for label being 1 to 4
print("sample class counts: ", sample_class_counts)

# Confusion matrix
conf_mat = np.zeros((C, C))
cumulative_loss = 0

for i in L: # Each decision option
    for j in L: # Each class label
        ind_ij = np.argwhere((decisions==i) & (labels==j))[:,0]
        cumulative_loss += Lambda[i,j]*len(ind_ij)
        if sample_class_counts[j] != 0:
            conf_mat[i, j] = round(len(ind_ij)/sample_class_counts[j], 3) # Average over class sample count
        else: 
            conf_mat[i, j] = 0
            
print("Confusion matrix:")
[print(conf_mat[row, nonzero_ix]) for row in nonzero_ix]

print("Average Loss MAP")
print(cumulative_loss/N)

print("Minimum Probability of Error:")
prob_error = 1 - np.diag(conf_mat).dot(sample_class_counts / N)
print(prob_error)

plt.title("Minimum Probability of Error Classified Sampled Data:  {:.3f}".format(prob_error))

# VISUALIZE w/ ARBITRARY FEATURES
marker_colors = 'rgbkymcrgbkymc' # Accomodates up to C=5
markers = 'oooooooxxxxxxx'

feature_pairs = [[10,2],[3,25],[210, 6]]
for feat in feature_pairs:
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    for l in L:
        ax.scatter(X[labels==l, feat[0]], X[labels==l, feat[1]], l/2, marker=markers[l], color=marker_colors[l])
    
    plt.xlabel('feature ' + str(feat[0]) + ':' + feature_names[feat[0]])
    plt.ylabel('feature ' + str(feat[1]) + ':' + feature_names[feat[1]])
    plt.title("Data projections to 2D space".format(F))
    ax.set_zlabel('true classes')
    ax.set_zticklabels([])


## PCA 

# First derive sample-based estimates of mean vector and covariance matrix:
mu_hat = np.mean(X, axis=0)
Sigma_hat = np.cov(X.T)

# Mean-subtraction is a necessary assumption for PCA, so perform this to obtain zero-mean sample set
C = X - mu_hat

# Get the eigenvectors (in U) and eigenvalues (in D) of the estimated covariance matrix
lambdas, U = np.linalg.eig(Sigma_hat)
# Get the indices from sorting lambdas in order of increasing value, with ::-1 slicing to then reverse order
idx = lambdas.argsort()[::-1]
print('top 5 features: ', feature_names[idx[0:5]])

# Extract corresponding sorted eigenvectors and eigenvalues
U = U[:, idx]
D = np.diag(lambdas[idx])

# Calculate the PC projections of zero-mean samples (in z)
Z = C.dot(U)

# Let's see what it looks like only along the first two PCs
fig = plt.figure(figsize=(10, 10))
plt.scatter(Z[:, 0], Z[:, 1])
plt.xlabel("z1")
plt.ylabel("z2")
plt.title("PCA projections to 2D space".format(F))

# What about 3D?
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='3d')

#ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], c='g', marker='o')

for l in L:
    ax.scatter(Z[labels==l, 0], Z[labels==l, 1], Z[labels==l, 2], marker=markers[l], color=marker_colors[l])

plt.legend()
plt.xlabel(r"$z_1$")
plt.ylabel(r"$z_2$")
ax.set_zlabel(r"$z_3$")
plt.title("PCA projections to 3D space")

# from sklearn.decomposition import PCA
# from sklearn.metrics import mean_squared_error

# # Max number of PCs based on rank of X, or min(n, N)
# X_rank = np.linalg.matrix_rank(X)

# rmse = np.zeros(X_rank)
# sum_eigenvals = np.zeros(X_rank)
# no_components = range(1, X_rank + 1)

# # Reconstruct the X data set from each set of projections
# for m in no_components:
#     # Reconstruct based on only the 'm' components (also revert mean-centering effect)
#     X_hat = Z[:, :m].dot(U[:, :m].T) + mu_hat
#     rmse[m-1] = np.sqrt(np.mean((X - X_hat)**2))
#     sum_eigenvals[m-1] = np.sum(D[:m])
    
# # Fraction of variance explained
# fraction_var = sum_eigenvals / np.trace(Sigma_hat)

# # MSE should be decreasing on each iteration, 0 for the nth
# fig = plt.figure(figsize=(10, 10))
# plt.plot(no_components, rmse)
# plt.xlabel("Dimension m of PCA")
# plt.ylabel("RMSE")

# # First eigenvalue should be significantly larger than the rest
# fig = plt.figure(figsize=(10, 10))
# plt.plot(no_components, sum_eigenvals)
# plt.xlabel("Dimension m of PCA")
# plt.ylabel("Sum of Eigenvalues")

# # About 95% variance explined is an acceptable target 
# fig = plt.figure(figsize=(10, 10))
# plt.plot(no_components, fraction_var)
# plt.xlabel("Dimension m of PCA")
# plt.ylabel("Fraction of Variance Explained")

plt.show()
