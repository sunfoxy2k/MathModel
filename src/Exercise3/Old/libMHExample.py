import math
import random
import numpy as np
import matplotlib.pyplot as plt
import moduleRK4SIR as RK4
import libMH as MH

size = int(input("Enter size of generated data: "))
givenBeta = float(input("Enter beta for generated data: "))
givenGamma = float(input("Enter gamma for generated data: "))

m = int(input("Enter m: "))

sigma1 = float(input("Enter sigma1: "))
sigma2 = float(input("Enter sigma2: "))

burnIn = int(input("Enter burnIn: "))


def makeXBetaGammaData(BETA, GAMMA, size):
    # X ~ gammaDistribution(BETA, GAMMA)
    # prob density function taken from https://en.wikipedia.org/wiki/Gamma_distribution   
    return np.random.gamma(BETA,1/GAMMA,size)

def makeDataSIR(BETA, GAMMA, size):
    sir_values = RK4.RK4SIR(10000, 1, 1, BETA, GAMMA, size-1, 1)
    x = []
    for d in sir_values:
        x.append(d.infected)
    return x

def likelihoodX(t, data):
    #log of the equation 15 in the assignment document
    n = len(data)
    return n*np.log(t[1]**t[0]/math.gamma(t[0])) + (t[0]-1)*np.sum(np.log(data)) - t[1]*np.sum(data)

def prior(t):
    # no preference / no known prior
    if(t[0] <= 0 or t[1] <= 0):
        return 0
    return 1


def normalProposal(t):
    # tPrime ~ Normal(t, sigma^2)
    #return [random.gauss(t[0], sigma1), random.gauss(t[1], sigma2)]
    return [np.random.normal(t[0],sigma1,1)[0],np.random.normal(t[1],sigma2,1)[0]]
normal_density = lambda sigma, mu, x :1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2))
def normalProposalPDF(tPrime, t):
    return normal_density(sigma1,t[0],tPrime[0]) * normal_density(sigma2,t[1],tPrime[1])

def gammaProposal(t):
    # tPrime ~ Gamma(t, sigma) sigma:scale
    #return [np.random.gamma(t[0],sigma1,1)[0],np.random.gamma(t[1],sigma2,1)[0]]
    return [np.random.gamma(t[0]**2/sigma1,sigma1/t[0],1)[0],np.random.gamma(t[1]**2/sigma2,sigma2/t[1],1)[0]]
gamma_density = lambda k, theta, x : x**(k-1)*np.exp(-x/theta) / (math.gamma(theta) * theta**k)
def gammaProposalPDF(tPrime, t):
    return gamma_density(t[0]**2/sigma1, sigma1/t[0], tPrime[0]) * gamma_density(t[1]**2/sigma2, sigma2/t[1], tPrime[1])

dummyPDF = lambda tPrime, t: 1 #we can use this instead of the proposalPDF if it's symmertric


data = makeXBetaGammaData(givenBeta, givenGamma, size)
#data = makeDataSIR(givenBeta, givenGamma, size)

trace, ac, re, iac, ire = MH.metropolisHasting(m, likelihoodX, prior, normalProposal, normalProposalPDF, data, [3, 1], burnIn)

sumg = 0
sumb = 0
betaTrace = []
gammaTrace = []
for b,g in trace:
    sumb += b
    sumg += g
    betaTrace.append(b)
    gammaTrace.append(g)

meanb = sumb/len(trace)
meang = sumg/len(trace)
posterior_data = np.random.gamma(meanb,1/meang,10000)

dataHist = plt.figure(figsize=(10,6))
ax = dataHist.add_subplot(1,2,1)
ax.hist( data,bins=50 ,)
ax.set_xlabel("Value")
ax.set_ylabel("Frequency")
ax.set_title("Figure 1: Histogram of dataset")

ax = dataHist.add_subplot(1,2,2)
ax.hist( posterior_data,bins=50 ,)
ax.set_xlabel("Value")
ax.set_title("Figure 2: predicted GammaDist.(beta, gamma)")


bTrace = plt.figure(figsize=(10,10))
ax = bTrace.add_subplot(2,2,1)
ax.plot(betaTrace)
ax.set_xlabel("Iteration")
ax.set_ylabel("Value")
ax.set_title("Figure 3: Trace of beta")

ax = bTrace.add_subplot(2,2,2)
ax.plot(gammaTrace)
ax.set_xlabel("Iteration")
ax.set_ylabel("Value")
ax.set_title("Figure 4: Trace of gamma")

ax = bTrace.add_subplot(2,2,3)
ax.hist(betaTrace , bins=50 ,)
ax.set_xlabel("Value")
ax.set_ylabel("Frequency")
ax.set_title("Figure 5: Histogram of beta")

ax = bTrace.add_subplot(2,2,4)
ax.hist(gammaTrace , bins=50 ,)
ax.set_xlabel("Value")
ax.set_ylabel("Frequency")
ax.set_title("Figure 6: Histogram of gamma")

betaAccepted = []
gammaAccepted= []
for b,g in ac:
    betaAccepted.append(b)
    gammaAccepted.append(g)

betaRejected = []
gammaRejected= []
for b,g in re:
    betaRejected.append(b)
    gammaRejected.append(g)

dataHist = plt.figure(figsize=(12,9))
ax = dataHist.add_subplot(2,1,1)
ax.plot(iac, betaAccepted, '+', color='blue', label="Accepted")
ax.plot(ire, betaRejected, 'x', color='red', label="Rejected")
ax.set_ylabel("Value")
ax.set_title("Figure 7: Sampling Plot of beta")
ax.legend()

ax = dataHist.add_subplot(2,1,2)
ax.plot(iac, gammaAccepted, '+', color='blue', label="Accepted")
ax.plot(ire, gammaRejected, 'x', color='red', label="Rejected")
ax.set_xlabel("Iteration")
ax.set_ylabel("Value")
ax.set_title("Figure 8: Sampling Plot of gamma")
ax.legend()


print(ac)
print("Acceptance Rate = ", len(ac)/(len(ac)+len(re)))

print("Expected b = ", meanb)
print("Expected g = ", meang)

print("Done", flush = True)


plt.show()

input()