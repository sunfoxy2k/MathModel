import Ex4Prior as pr
import Ex3libMHSimplified as MH
import csv
import numpy as np
import math
import scipy.special as sp

#calling ex3 sampling function for prior of beta and gamma, the sample is stored in the list 'trace'
m = 5000#int(input("Enter m: "))
burnIn = 2000#int(input("Enter burnIn: "))
trace, ac, re, iac, ire = MH.metropolisHasting(m, pr.logPDF, pr.normalProposal, pr.dummyPDF, [1, 0.1], burnIn)

#extracting and reprocess csv, our MH model use week, not day, also, the data is in culmulative form of gamma distribution
with open('cumumlativedaily_confirmed.csv','rt')as f:
  fileContent = csv.reader(f)
  data1 = []
  for row in fileContent:
      data1.append(row)
with open('cumumlativedaily_death.csv','rt')as f:
  fileContent = csv.reader(f)
  data2 = []
  for row in fileContent:
      data2.append(row)
with open('cumumlativedaily_recovered.csv','rt')as f:
  fileContent = csv.reader(f)
  data3 = []
  for row in fileContent:
      data3.append(row)

numberOfDay = len(data1) - 1
numberOfWeek = int(numberOfDay / 7)
numberOfCountry = len(data1[0]) - 1 #will crash if data is empty

countryList = []
for i in range (0, numberOfCountry):
    countryList.append(data1[0][i+1])

offset = 0

IRList = [[0 for x in range(numberOfDay-offset)] for y in range(numberOfCountry)]
for i in range (0, numberOfCountry):
    for j in range (0, numberOfDay-offset):
        IRList[i][j] = int(data1[j+1+offset][i+1])
IRList = np.array(IRList) 

RList = [[0 for x in range(numberOfDay-offset)] for y in range(numberOfCountry)]
for i in range (0, numberOfCountry):
    for j in range (0, numberOfDay-offset):
        RList[i][j] = int(data2[j+1+offset][i+1]) + int(data3[j+1+offset][i+1])
RList = np.array(RList)


offset += 1        
IList = [[0 for x in range(numberOfDay-offset)] for y in range(numberOfCountry)]
for i in range (0, numberOfCountry):
    for j in range (0, numberOfDay-offset):
        IList[i][j] = IRList[i][j] - RList[i][j]
IList = np.array(IList)

dIdRList = [[0 for x in range(numberOfDay-offset)] for y in range(numberOfCountry)]
for i in range (0, numberOfCountry):
    for j in range (0, numberOfDay-offset):
        dIdRList[i][j] = IRList[i][j+1] - IRList[i][j]
        if dIdRList[i][j] <= 0 : dIdRList[i][j] = 0.0001
dIdRList = np.array(dIdRList)

dRList = [[0 for x in range(numberOfDay-offset)] for y in range(numberOfCountry)]
for i in range (0, numberOfCountry):
    for j in range (0, numberOfDay-offset):
        dRList[i][j] = RList[i][j+1] - RList[i][j]
        if dRList[i][j] <= 0 : dRList[i][j] = 0.0001
dRList = np.array(dRList)
#csv part done

def loglikelihood_standard_normal(X): #for a continuous distribution, this gave a very very low number
    n = len(X)
    return -1/2*np.sum(np.power(X,2))-n*np.log(np.sqrt(2*np.pi))

bound = 0.5 #confidence range, should be positive
def loglikelihood_standard_normal_accept_ratio(X):
    return np.log(((X > -bound) & (X < bound )).sum() / len(X))

def loglikelihoodXinGammaDist(beta, gamma, X):#pdf of gammaDist, we don't use this, it's very imprecise
    n = len(X)
    return n*beta*np.log(gamma) - n*np.log(sp.gamma(beta)) + (beta-1)*np.sum(np.log(X)) - gamma*np.sum(X)

ContryOffset = [0,0,10,0,0,0,0,0,38,0,0,0,0,0]
gammaDelay =   [0,0, 0,0,0,7,7,0, 0,0,0,0,0,0]
betaDelay =    [0,0, 0,0,0,7,7,0, 0,0,0,0,0,0]

R0List = []
max = -999999
for i in range (0, numberOfCountry, 1):
    dRL = []
    dIdRL = []
    IListGamma = []
    IListBeta = []
    maxDelay = gammaDelay[i] if gammaDelay[i] > betaDelay[i] else betaDelay[i]
    for j in range(ContryOffset[i],len(dRList[i])-maxDelay):
        dRL.append(dRList[i][j+gammaDelay[i]])
        dIdRL.append(dIdRList[i][j+betaDelay[i]])
        IListGamma.append(IList[i][j])
        IListBeta.append(IList[i][j])
    
    dRL = np.array(dRL)
    dIdRL = np.array(dIdRL)
    IListGamma = np.array(IListGamma)
    IListBeta = np.array(IListBeta)

    sum = 0
    for beta, gamma in trace:
        lamdaGamma = (gamma*IListGamma)
        #likelihoodGamma = loglikelihood_standard_normal((dRL-lamdaGamma) / np.sqrt(lamdaGamma))
        likelihoodGamma = loglikelihood_standard_normal_accept_ratio((dRL-lamdaGamma) / np.sqrt(lamdaGamma))
        

        lamdaBeta = (beta*IListBeta)
        #likelihoodBeta = loglikelihood_standard_normal((dIdRL-lamdaBeta) / np.sqrt(lamdaBeta))
        likelihoodBeta = loglikelihood_standard_normal_accept_ratio((dIdRL-lamdaBeta) / np.sqrt(lamdaBeta))

        likelihood = likelihoodGamma + likelihoodBeta
        if likelihood > max:
            max = likelihood
            #print("new max: ", likelihood)

        sum += np.exp(likelihood) * beta/gamma
    R0List.append(sum)

for i in range (0, numberOfCountry, 1):
    print(countryList[i], ": R0 = ", R0List[i])

print("Done", flush = True)