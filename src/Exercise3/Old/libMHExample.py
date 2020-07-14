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

dummyPDF = lambda tPrime, t: 1


data = makeXBetaGammaData(givenBeta, givenGamma, size)
#data = makeDataSIR(givenBeta, givenGamma, size)

ac, re, trace = MH.metropolisHasting(m, likelihoodX, prior, normalProposal, normalProposalPDF, data, [3, 1], burnIn,0)

sumg = 0
sumb = 0
for b,g in ac:
    sumb += b
    sumg += g
meanb = sumb/len(ac)
meang = sumb/len(ac)
posterior_data = np.random.gamma(meanb,1/g,10000)

dataHist = plt.figure(figsize=(10,6))
ax = dataHist.add_subplot(1,2,1)
ax.hist( data,bins=35 ,)
ax.set_xlabel("Value")
ax.set_ylabel("Frequency")
ax.set_title("Figure 1: Histogram of dataset")

ax = dataHist.add_subplot(1,2,2)
ax.hist( posterior_data,bins=35 ,)
ax.set_xlabel("Value")
ax.set_title("Figure 2: predicted GammaDist.(beta, gamma)")

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
ax.set_title("Figure 3: Trace of beta")

ax = bTrace.add_subplot(2,2,2)
ax.plot(gammaTrace)
ax.set_xlabel("Iteration")
ax.set_ylabel("Value")
ax.set_title("Figure 4: Trace of gamma")

betaAccepted = []
gammaAccepted= []
for b,g in ac:
    betaAccepted.append(b)
    gammaAccepted.append(g)

ax = bTrace.add_subplot(2,2,3)
ax.hist(betaAccepted, bins=35 ,)
ax.set_xlabel("Value")
ax.set_ylabel("Frequency")
ax.set_title("Figure 5: Histogram of beta")

ax = bTrace.add_subplot(2,2,4)
ax.hist(gammaAccepted, bins=35 ,)
ax.set_xlabel("Value")
ax.set_ylabel("Frequency")
ax.set_title("Figure 6: Histogram of gamma")


print(ac)
print("Acceptance Rate = ", len(ac)/(len(ac)+len(re)))

print("Expected b = ", meanb)
print("Expected g = ", meang)

print("Done", flush = True)


plt.show()

input()