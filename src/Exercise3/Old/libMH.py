import numpy as np

def metropolisHasting(n, likelihood, prior, proposal, proposalPDF, data, t0, burnIn):   
    #  n: number of iteration
    # likelihood(t, data): returns log of the likelihood of how much the parameter t fit the data 
    # prior(t): prior belief, restriction, preference for the parameter t
    # proposal(t) = q(t) : return a new t' from current t
    # proposalPDF(tPrime, t) = q(tPrime | t) : the probability density function of q(t)
    # data: data from where we wish to approximate the parameter which model the data's population
    # t0: inital parameter
    # burnIn: number of ignored sample

    #Return:
    #traceValue: The return trace of t at all iteration, this will be our returned sample
    #accepted: The accepted values, used for plotting
    #rejected: The rejected values, used for plotting
    #samplingPlotAc: The accepted values's iteration, used for plotting
    #samplingPlotRe: The rejected values's iteration, used for plotting

    t = t0;
    traceValue = []
    accepted = []
    rejected = []  
    samplingPlotAc = []
    samplingPlotRe = []
    
    for j in range(burnIn):
            tPrime = proposal(t)

            p = likelihood(t, data) + np.log(prior(t)) + np.log(proposalPDF(tPrime, t))
            pPrime = likelihood(tPrime, data) + np.log(prior(tPrime)) + np.log(proposalPDF(t, tPrime))
     
            if pPrime > p:              
                t = tPrime
            else:
                a = np.random.uniform(0, 1)
                if a < np.exp(pPrime - p):
                    t = tPrime

    for i in range(n):
            tPrime = proposal(t)

            p = likelihood(t, data) + np.log(prior(t)) + np.log(proposalPDF(tPrime, t))
            pPrime = likelihood(tPrime, data) + np.log(prior(tPrime)) + np.log(proposalPDF(t, tPrime))
     
            if pPrime > p:              
                t = tPrime
                accepted.append(t)
                traceValue.append(t)
                samplingPlotAc.append(i)
            else:
                a = np.random.uniform(0, 1)
                if a < np.exp(pPrime - p):
                    t = tPrime
                    accepted.append(t)
                    traceValue.append(t)
                    samplingPlotAc.append(i)
                else:
                    rejected.append(tPrime) 
                    traceValue.append(t)
                    samplingPlotRe.append(i)

    return  traceValue, accepted, rejected, samplingPlotAc, samplingPlotRe
