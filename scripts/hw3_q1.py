from matplotlib import projections
from hw3_q1_functions import *
import matplotlib.pyplot as plt

trainingSamples = [100, 200, 500, 1000, 2000, 5000]
validationSamples = 100000
trainingData = []

priors = [0.25, 0.25, 0.25, 0.25]

mu = np.array( [[2, 2, 2],  
                [1, 2, 3],
                [6, 5, 4],
                [4, 1, 5]]) # distribution means

Sigma = np.array([[[1, -0.5, 0.3],
                   [-0.5, 1, -0.5],
                   [0.3, -0.5, 1]],
                  [[1, 0.4, 0.1],
                   [0.4, 1, 0.4],
                   [0.1, 0.4, 1]],
                    2*np.eye(3),
                  [[1, 0.1, 0.3],
                   [0.1, 1, 0.1],
                   [0.3, 0.1, 1]]]) # distribution covariances

# some counting/size variables
n = mu.shape[1]
C = len(priors) # how MANY classes there are
L = np.array(range(C)) # Labels 0,1,2,3 like we want

for N in trainingSamples:
    trainingData.append(generate_data(mu, Sigma, priors, N))

validationData = generate_data(mu, Sigma, priors, validationSamples)

# visualize the data for each training set:
visualize_data(trainingData, trainingSamples, L, "Training", verbose=False)
sampleClassCounts = visualize_data([validationData], [validationSamples], L, "Validation")[0]

# Esimtate Theoretical MPE: 
# achieved with 0-1 loss, which makes the loss matrix:

Lambda_MAP = np.ones([4,4])
np.fill_diagonal(Lambda_MAP,0)

# and the matrix form of the loss minimization decision rule is
# Risk across all decisions = loss matrix * (class likelihoods * class priors) = loss * pirors 

# [[ R(D(x) = 1 | x) ],            [[P(Y = 1 | x) 0 0 0 ],    [[p(x | Y = 1)],
#  [ R(D(x) = 2 | x) ],             [0 P(Y = 2 | x) 0 0 ],     [p(x | Y = 2)],
#  [ R(D(x) = 3 | x) ], = Lambda *  [0 0 P(Y = 3 | x) 0 ],  *  [p(x | Y = 3)], 
#  [ R(D(x) = 4 | x) ]]             [0 0 0 P(Y = 4 | x) ]]     [p(x | Y = 4)]]

# Calculate class-conditional likelihoods p(x|Y=j) for each label of the N observations
class_cond_likelihoods = np.array([multivariate_normal.pdf(validationData[0], mu[l], Sigma[l]) for l in L])
class_priors = np.diag(priors) # DO WE USE REAL PRIORS OR THE COUNTED PRIORS??
print(class_cond_likelihoods.shape)
print(class_priors.shape)
class_posteriors = class_priors.dot(class_cond_likelihoods)

# We want to create the risk matrix of size 4 x N 
cond_risk = Lambda_MAP.dot(class_posteriors)
print(cond_risk.shape)

# Get the decision for each column in risk_mat
decisions = np.argmin(cond_risk, axis=0)
print(decisions.shape)

# Confusion matrix
conf_mat = np.zeros((C, C))

cumulative_loss = 0
for i in L: # Each decision option
    for j in L: # Each class label
        ind_ij = np.argwhere((decisions==i) & (validationData[1]==j))
        cumulative_loss += Lambda_MAP[i,j]*len(ind_ij)
        conf_mat[i, j] = round(len(ind_ij)/sampleClassCounts[j],3) # Average over class sample count       

print("Confusion matrix:")
print(conf_mat)

print("Average Loss MAP")
print(cumulative_loss/validationSamples)

print("Minimum Probability of Error:")
prob_error = 1 - np.diag(conf_mat).dot(sampleClassCounts / validationSamples)
print(prob_error)