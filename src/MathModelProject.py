import MH
import internetExample as ie
import RK4
import math
import random
import numpy as np

#accepted, rejected = ie.metropolis_hastings(ie.manual_log_like_normal,ie.prior,ie.transition_model,[ie.mu_obs,0.1], 50000,ie.observation,ie.acceptance)
#print(accepted)

def makeXBetaGammaData(BETA, GAMMA, size):
    # X ~ gammaDistribution(BETA, GAMMA)
    # prob density function taken from https://en.wikipedia.org/wiki/Gamma_distribution
    N = 10000
    INIT_INFECTED = 1
    INIT_RECOVERED = 0
    TIME = size-1
    #dataSIR = RK4.RK4SIR(N, INIT_INFECTED, INIT_RECOVERED, BETA, GAMMA, TIME, 1)

    X = []
    #for d in dataSIR:
    #    X.append(d.recovered + d.infected)
    for i in range(size):
        X.append(random.gammavariate(BETA,GAMMA))
    return X



def likelihoodX(t, data):
    #log of the equation 15 in the assignment document
    
    if(t[0] <= 0 or t[1] <= 0):
       return 0

    lnGammaPowBeta = t[0] * math.log(t[1])
    lnGammaFunctBeta = math.gamma(t[0]) 

    result = len(data) * (lnGammaPowBeta - lnGammaFunctBeta)

    sumLnX = 0
    sumX = 0
    for x in data:
        #result = result * (t[1] ** t[0]) / math.gamma(t[0]) * (x ** (t[0] - 1)) * math.exp(-t[1] * x)
        sumLnX += math.log(x)
        sumX += x

    result = result + (t[0] - 1) * sumLnX - t[1] * sumX
    return result

def prior(t):
    # no preference / no known prior
    if(t[0] <= 0 or t[1] <= 0):
        return 0
    return 1

sigma1 = 1
sigma2 = 0.5
def proposal(t):
    # tPrime ~ Normal(t, sigma^2)

    #not sure this is right?
    return [random.gauss(t[0], sigma1), random.gauss(t[1], sigma2)]

def proposalPDF(tPrime, t):
    # tPrime ~ Normal(t, sigman^2)
    # Let y = (tPrime - t)/sigma
    # => y ~ N(0, 1)
    # f(Y = y) = e^(-y^2/2) / sqrt(2*pi)

    y = [(tPrime[0] - t[0])/sigma, (tPrime[1] - t[1])/sigma]
    probBeta = math.exp(- y[0]**y[0] / 2) / math.sqrt(2 * math.pi)
    probGamma = math.exp(- y[1]**y[1] / 2) / math.sqrt(2 * math.pi)

    #something might be wrong with this, joint normal probability?
    return probBeta * probGamma 

size = 3000
data = makeXBetaGammaData(2, 0.5, size)

sample = 10000
#ac, re = MH.metropolisHasting(sample, 1, 1, 3, ie.manual_log_like_normal, ie.prior, ie.transition_model, proposalPDF, ie.observation, [ie.mu_obs, 0.1])
ac, re = MH.metropolisHasting(sample, 1, 1, 3, likelihoodX, prior, proposal, proposalPDF, data, [1, 0.1])

print(ac)
print("Acceptance Rate = ", len(ac)/(len(ac)+len(re)))
#print(re)

input()
print("Done")