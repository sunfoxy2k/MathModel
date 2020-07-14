import numpy as np

def metropolisHasting(m, logPdf, proposal, logProposalPDF, t0, burnIn = 0):   
    #  m: number of sample to draw
    # logPdf: h(t)
    # proposal(t) = q(t) : return a new t' from current t
    # proposalPDF(tPrime, t) = q(tPrime | t) : the probability density function of q(t)
    # t0: inital parameter 
    # burnIn: number of ignored first iteration

    ###
    # The function document need a general explanation 
    # Give a return format 
    # The paramenter need a data type and limitation 
    ###

    t = t0
    i = 0

    ###
    # list is fine but np.array is better
    # list in python require more memory than np.array
    # np.array tend to have a fix-size 
    # you could convert list to array, at the end.
    ###
    accepted = []
    rejected = []
    iterateAC = [] # ThienPhuc: The name is hard to understand???

    samplingPlotAc = []
    samplingPlotRe = []

    ###
    # Why coding you consider two things: Time Complexity and Space Complexity
    # I think you are trying to save time when ploting 
    # You can improve both time and space here
    # create one array to store all t over the iteration, dtype will be float
    # create one array to store the iteration that is accepted
    ###

    for j in range(burnIn):
        ###
        # this code seems duplicate with below 
        # you should create a private function here
        # It helps a lot with the readability
        ###
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
        # ThienPhuc: should create a local variable as counting instead
         tPrime = proposal(t)

         p = logPdf(t) +logProposalPDF(tPrime, t)
         pPrime = logPdf(tPrime) + logProposalPDF(t, tPrime)
     
         if pPrime > p:
             t = tPrime
             ###
             # A lot of duplicate codes you could try to reduce your code 
             # make it easier to update by create a private functio
             ###
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
