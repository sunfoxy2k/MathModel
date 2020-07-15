import Ex4Prior as pr
import Ex3libMHSimplified as MH
import csv
import numpy as np
import math
import scipy.special as sp

#calling ex3 sampling function for prior of beta and gamma, the sample is stored in the list 'trace'
m = int(input("Enter m: "))
burnIn = int(input("Enter burnIn: "))
trace, ac, re, iac, ire = MH.metropolisHasting(m, pr.logPDF, pr.normalProposal, pr.logNormalProposalPDF, [0.5, 2], burnIn)

#extracting and reprocess csv, our MH model use week, not day, also, the data is in culmulative form of gamma distribution
with open('confirmed.csv','rt')as f:
  fileContent = csv.reader(f)
  data = []
  for row in fileContent:
      data.append(row)
numberOfDay = len(data) - 1
numberOfWeek = int(numberOfDay / 7)
numberOfCountry = len(data[0]) - 1 #will crash if data is empty

countryList = []
for i in range (0, numberOfCountry):
    countryList.append(data[0][i+1])

    ##daily case
confirmedCaseDailyList = [[0 for x in range(numberOfDay)] for y in range(numberOfCountry)]
for i in range (0, numberOfCountry):
    for j in range (0, numberOfDay):
        confirmedCaseDailyList[i][j] = int(data[j+1][i+1])
        if confirmedCaseDailyList[i][j] == 0: confirmedCaseDailyList[i][j] = 0.0001
    ##daily case

    ##weekly culmulative
confirmedCaseList = [[0 for x in range(numberOfWeek)] for y in range(numberOfCountry)] #2d array comprehension?
for i in range (0, numberOfCountry):
    for j in range (0, numberOfWeek):
        startDay = j*7 + 1
        sumCaseOfWeek = 0
        for k in range(startDay, startDay + 7):
            sumCaseOfWeek += int(data[k][i+1])
        confirmedCaseList[i][j] = sumCaseOfWeek
        if confirmedCaseList[i][j] == 0: confirmedCaseList[i][j] = 0.0001
    ##END weekly culmulative

#i ~ gammaDist(beta, gamma) (gamma is the rate parameter, not scale)
infectedWeeklyList = [[0 for x in range(numberOfWeek-1)] for y in range(numberOfCountry)]
for i in range (0, numberOfCountry):
    for j in range (0, numberOfWeek-1):
        infectedWeeklyList[i][j] = confirmedCaseList[i][j+1] - confirmedCaseList[i][j]
        if infectedWeeklyList[i][j] == 0: infectedWeeklyList[i][j] = 0.0001 
#csv part done
#print(infectedWeeklyList)
#print(confirmedCaseDailyList)

def loglikelihoodXinGammaDist(beta, gamma, X):
    n = len(X)
    return n*np.log(gamma**beta/math.gamma(beta)) + (beta-1)*np.sum(np.log(X)) - gamma*np.sum(X)

gamma_culmulative = lambda a, b, x: sp.gammainc(a, b*x)/sp.gamma(a)
def loglikelihoodXinCulmulativeGammaDist(beta, gamma, X):
    sum = 0
    for xi in X:
        sum += np.log(gamma_culmulative(beta, gamma, xi))
    return sum

R0List = []
for i in range (0, numberOfCountry, 1):
    sum = 0
    for beta, gamma in trace:
        #likelihood = loglikelihoodXinGammaDist(beta, gamma, infectedWeeklyList[i])
        likelihood = loglikelihoodXinCulmulativeGammaDist(beta, gamma, confirmedCaseList[i])
        #likelihood = loglikelihoodXinCulmulativeGammaDist(beta, gamma, confirmedCaseDailyList[i])
        #print(likelihood)
        sum += np.exp(likelihood) * beta/gamma
    R0List.append(sum)

for i in range (0, numberOfCountry, 1):
    print(countryList[i], ": R0 = ", R0List[i])

print("Done", flush = True)