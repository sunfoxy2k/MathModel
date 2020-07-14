import numpy as np

def metropolisHasting(n, likelihood, prior, proposal, proposalPDF, data, t0, burnIn, mode = 0):   
    #  n: usage depend on the mode
    # mode =
    #   0: loop n time to take sample
    #   1: find n accepted sameples 
    # likelihood(t, data): returns log of the likelihood of how much the parameter t fit the data 
    # prior(t): prior belief, restriction, preference for the parameter t
    # proposal(t) = q(t) : return a new t' from current t
    # proposalPDF(tPrime, t) = q(tPrime | t) : the probability density function of q(t)
    # data: data from where we wish to approximate the parameter which model the data's population
    # t0: inital parameter
    # burnIn: number of ignored sample

    t = t0;
    accepted = []
    rejected = []
    iterateAC = []
    
    for i in range(burnIn):
            tPrime = proposal(t)

            p = likelihood(t, data) + np.log(prior(t)) + np.log(proposalPDF(tPrime, t))
            pPrime = likelihood(tPrime, data) + np.log(prior(tPrime)) + np.log(proposalPDF(t, tPrime))
     
            if pPrime > p:              
                t = tPrime
            else:
                a = np.random.uniform(0, 1)
                if a < np.exp(pPrime - p):
                    t = tPrime

    if mode == 0:
        for i in range(n):
            tPrime = proposal(t)

            p = likelihood(t, data) + np.log(prior(t)) + np.log(proposalPDF(tPrime, t))
            pPrime = likelihood(tPrime, data) + np.log(prior(tPrime)) + np.log(proposalPDF(t, tPrime))
     
            if pPrime > p:              
                t = tPrime
                accepted.append(t)
                iterateAC.append(t)
            else:
                a = np.random.uniform(0, 1)
                if a < np.exp(pPrime - p):
                    t = tPrime
                    accepted.append(t)
                    iterateAC.append(t)
                else:
                    rejected.append(tPrime) 
                    iterateAC.append(t)
    elif mode == 1:
        while len(accepted) < n:
            tPrime = proposal(t)

            p = likelihood(t, data) + np.log(prior(t)) + np.log(proposalPDF(tPrime, t))
            pPrime = likelihood(tPrime, data) + np.log(prior(tPrime)) + np.log(proposalPDF(t, tPrime))
     
            if pPrime > p:
                t = tPrime
                accepted.append(t)
                iterateAC.append(t)                
            else:
                a = np.random.uniform(0, 1)
                if a < np.exp(pPrime - p):
                    t = tPrime
                    accepted.append(t)
                    iterateAC.append(t)
                else:
                    rejected.append(tPrime)
                    iterateAC.append(t)
                    
    return accepted, rejected, iterateAC
