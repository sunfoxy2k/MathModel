import numpy as np

def metropolisHasting(m, logPdf, proposal, logProposalPDF, t0, burnIn = 0):   
    #  m: number of sample to draw
    # logPdf: h(t)
    # proposal(t) = q(t) : return a new t' from current t
    # proposalPDF(tPrime, t) = q(tPrime | t) : the probability density function of q(t)
    # t0: inital parameter
    # burnIn: number of ignored first iteration

    t = t0;
    accepted = []
    rejected = []
    iterateAC = []
    samplingPlotAc = []
    samplingPlotRe = []
    i = 0

    for j in range(burnIn):
            tPrime = proposal(t)

            p = logPdf(t) +logProposalPDF(tPrime, t)
            pPrime = logPdf(tPrime) + logProposalPDF(t, tPrime)
     
            if pPrime > p:              
                t = tPrime
            else:
                a = np.random.uniform(0, 1)
                if a < np.exp(pPrime - p):
                    t = tPrime

    while len(accepted) < m:
         tPrime = proposal(t)

         p = logPdf(t) +logProposalPDF(tPrime, t)
         pPrime = logPdf(tPrime) + logProposalPDF(t, tPrime)
     
         if pPrime > p:
             t = tPrime
             accepted.append(t)
             iterateAC.append(t)
             samplingPlotAc.append(i)
         else:
             a = np.random.uniform(0, 1)
             if a < np.exp(pPrime - p):
                 t = tPrime
                 accepted.append(t)
                 iterateAC.append(t)
                 samplingPlotAc.append(i)
             else:
                 rejected.append(tPrime)
                 iterateAC.append(t)
                 samplingPlotRe.append(i)
         i += 1                   
    return accepted, rejected, iterateAC, samplingPlotAc, samplingPlotRe