import libMH as MH
import math
import random
import numpy as np
import matplotlib.pyplot as plt

def makeXBetaGammaData(BETA, GAMMA, size):
    # X ~ gammaDistribution(BETA, GAMMA)
    # prob density function taken from https://en.wikipedia.org/wiki/Gamma_distribution   
    return np.random.gamma(BETA,1/GAMMA,size)

def likelihoodX(t, data):
    #log of the equation 15 in the assignment document
    n = len(data)
    return n*np.log(t[1]**t[0]/math.gamma(t[0])) + (t[0]-1)*np.sum(np.log(data)) - t[1]*np.sum(data)

def prior(t):
    # no preference / no known prior
    if(t[0] <= 0 or t[1] <= 0):
        return 0
    return 1

sigma1 = 0.01
sigma2 = 0.01
def proposal(t):
    # tPrime ~ Normal(t, sigma^2)
    #return [random.gauss(t[0], sigma1), random.gauss(t[1], sigma2)]
    return [np.random.normal(t[0],sigma1,1)[0],np.random.normal(t[1],sigma2,1)[0]]

def proposalPDF(tPrime, t):
    return normal_density(sigma1,t[0],tPrime[0]) * normal_density(sigma2,t[1],tPrime[1])

normal_density = lambda sigma, mu, x :1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2))

dummyPDF = lambda tPrime, t: 1

size = 30000
data = makeXBetaGammaData(5, 2, size)

dataHist = plt.figure(figsize=(6,6))
ax = dataHist.add_subplot(1,1,1)
ax.hist( data,bins=35 ,)
ax.set_xlabel("Value")
ax.set_ylabel("Frequency")
ax.set_title("Figure 1: Histogram of dataset")

sample = 10000
ac, re, trace = MH.metropolisHasting(sample, likelihoodX, prior, proposal, proposalPDF, data, [7, 1], 0)


betaTrace = []
gammaTrace = []
for b,g in trace:
    betaTrace.append(b)
    gammaTrace.append(g)

bTrace = plt.figure(figsize=(10,10))
ax = bTrace.add_subplot(2,2,1)
ax.plot(betaTrace)
ax.set_xlabel("Iteration")
ax.set_ylabel("Value")
ax.set_title("Figure 2: Trace of beta")

ax = bTrace.add_subplot(2,2,2)
ax.plot(gammaTrace)
ax.set_xlabel("Iteration")
ax.set_ylabel("Value")
ax.set_title("Figure 3: Trace of gamma")

betaAccepted = []
gammaAccepted= []
for b,g in ac:
    betaAccepted.append(b)
    gammaAccepted.append(g)

ax = bTrace.add_subplot(2,2,3)
ax.hist(betaAccepted, bins=35 ,)
ax.set_xlabel("Value")
ax.set_ylabel("Frequency")
ax.set_title("Figure 4: Histogram of beta")

ax = bTrace.add_subplot(2,2,4)
ax.hist(gammaAccepted, bins=35 ,)
ax.set_xlabel("Value")
ax.set_ylabel("Frequency")
ax.set_title("Figure 5: Histogram of gamma")


print(ac)
print("Acceptance Rate = ", len(ac)/(len(ac)+len(re)))

sumg = 0
sumb = 0
for b,g in ac:
    sumb += b
    sumg += g
print("Expected b = ", sumb/len(ac))
print("Expected g = ", sumg/len(ac))

print("Done", flush = True)


plt.show()

input()