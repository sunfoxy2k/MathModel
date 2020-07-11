
import random
import math

def metropolisHasting(n, N, acceptanceRate, mode, likelihood, prior, proposal, proposalPDF, data, t0):
    # n: usage depend on the mode
    # N: used for mode 0
    # acceptanceRate : used for mode 0
    # mode =
    #   0: using normal distribution as proposal, adjusted varience to follow an inital acceptanceRate after N step, retun n accepted samples (burn-in)
    #   1: loop n time to take sample
    #   2: find n accepted sameples 
    #   3: same as 0 but assume symetric proposal
    #   4: same as 1 but assume symetric proposal
    # likelihood(t, data): returns the likelihood of how much the parameter t fit the data 
    # prior(t): prior belief, restriction, preference for the parameter t
    # proposal(t) = q(t) : return a new t' from current t (random walk)
    # proposalPDF(tPrime, t) = q(tPrime | t) : the probability density function of q(t), used for mode 0,1,2
    # data: data from where we wish to approximate the parameter which model the data's population
    # t0: inital parameter

    t = t0;
    accepted = []
    rejected = []
    
    if mode == 0:
        print("not implemented yet.")
    elif mode == 1:
        for i in range(n):
            tPrior = prior(t)
            tLikely = likelihood(t, data)
            tPropose = proposalPDF(tPrime, t)

            tPrime = proposal(t)
            tPrimePrior = prior(tPrime)
            tPrimeLikely = likelihood(tPrime, data)
            tPrimePropose = proposalPDF(t, tPrime)

            r = (tPrimeLikely * tPrimePrior * tPrimePropose) / (tLikely * tPrior * tPropose)
            
            if r >= 1:
                accepted.append(tPrime)
                t = tPrime
            else:
                a = random.uniform(0, 1)
                if a < r:
                     accepted.append(tPrime)
                     t = tPrime
                else:
                    rejected.append(tPrime)     
    elif mode == 2:
        while len(accepted) < n:
            tPrior = prior(t)
            tLikely = likelihood(t, data)
            tPropose = proposalPDF(tPrime, t)

            tPrime = proposal(t)
            tPrimePrior = prior(tPrime)
            tPrimeLikely = likelihood(tPrime, data)
            tPrimePropose = proposalPDF(t, tPrime)

            r = (tPrimeLikely * tPrimePrior * tPrimePropose) / (tLikely * tPrior * tPropose)
            
            if r >= 1:
                accepted.append(tPrime)
                t = tPrime
            else:
                a = random.uniform(0, 1)
                if a < r:
                     accepted.append(tPrime)
                     t = tPrime
                else:
                    rejected.append(tPrime)   
    elif mode == 3:
        for i in range(n):
            tPrior = prior(t)
            tLikely = likelihood(t, data)

            tPrime = proposal(t)
            tPrimePrior = prior(tPrime)
            tPrimeLikely = likelihood(tPrime, data)
            
            pPrime = tPrimeLikely + math.log(tPrimePrior)
            p = tLikely + math.log(tPrior)
     
            if pPrime > p:
                accepted.append(tPrime)
                t = tPrime
            else:
                a = random.uniform(0, 1)
                if a < math.exp(pPrime - p):
                     accepted.append(tPrime)
                     t = tPrime
                else:
                    rejected.append(tPrime)
    elif mode == 4:
        while len(accepted) < n:
            tPrior = prior(t)
            tLikely = likelihood(t, data)

            tPrime = proposal(t)
            tPrimePrior = prior(tPrime)
            tPrimeLikely = likelihood(tPrime, data)

            r = (tPrimeLikely * tPrimePrior) / (tLikely * tPrior)
            
            if r >= 1:
                accepted.append(tPrime)
                t = tPrime
            else:
                a = random.uniform(0, 1)
                if a < r:
                     accepted.append(tPrime)
                     t = tPrime
                else:
                    rejected.append(tPrime)

    return accepted, rejected
