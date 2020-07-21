import numpy as np
import moduleRK4SIR as rk4
import csv

def traniningLoss(tolerance, lossFuntion, proposal, data, t0, tolTest=100):   
    # tolerance: if ratio of change in loss value smaller than this, we start counting
    # lossFuntion: L(t, data): loss value of predicted data base on t and real data, it's up to you to decide the prediction outside
    # proposal(t) = q(t) : return a new t' from current t, should be sysmetric
    # t0: inital parameter
    # tolTest: number of check for delta < tolerance

    #Return:
    #t: final [beta, gamma]

    t = t0;

    insideTol = 0
    loss = lossFuntion(t, data)
    while insideTol < tolTest:
        tNew = proposal(t)
        newLoss = lossFuntion(tNew, data)
        
        r = loss/newLoss
        if(tNew[0] <= 0 or tNew[1] <= 0): r = 0

        a = np.random.uniform(0, 1)
        if a <= r:
            print(insideTol, int(loss), t)
            t = tNew
            loss = newLoss
            if(np.abs(1-r) < tolerance):
                insideTol += 1
            else:
                insideTol = 0
    return t, loss

def SIRlossFunction(t, data):
    predictI, predictS = rk4.RK4SIR(1000000000, I0, R0, t[0], t[1], time, 1)
    #dataI = np.array(data[0])
    #dataS = np.array(data[1])
    return (meanSquareError(data[0], predictI) + meanSquareError(data[1], predictS))/2

def meanSquareError(data, prediction):
    return np.sum(np.power(data - prediction, 2)) / len(data)

def normalProposal(t):
    # tPrime ~ Normal(t, sigma)
    return [np.random.normal(t[0],0.1,1)[0],np.random.normal(t[1],0.05,1)[0]]


#extracting and reprocess csv
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
    #IRList[i] = np.array(IRList[i])
IRList = np.array(IRList) 

RList = [[0 for x in range(numberOfDay-offset)] for y in range(numberOfCountry)]
for i in range (0, numberOfCountry):
    for j in range (0, numberOfDay-offset):
        RList[i][j] = int(data2[j+1+offset][i+1]) + int(data3[j+1+offset][i+1])
    #RList[i] = np.array(RList[i])
RList = np.array(RList)
        
IList = [[0 for x in range(numberOfDay-offset)] for y in range(numberOfCountry)]
for i in range (0, numberOfCountry):
    for j in range (0, numberOfDay-offset):
        IList[i][j] = IRList[i][j] - RList[i][j]
    #IList[i] = np.array(IList[i])
IList = np.array(IList)


#csv part done

print("Country List:")
count = 0
for s in countryList:
    print(count,". ",s)
    count += 1

countryID = int(input("Choose a country: "))
start = int(input("Choose start day index (0-168): "))
end = int(input("Choose end day index (startDay -> 168): "))

data = []
data.append(np.array(IList[countryID])[start:end])
data.append(np.array(RList[countryID])[start:end])

I0 = data[0][start]
R0 = data[1][start]
time = end - start - 1

betaGamma, loss = traniningLoss(0.05, SIRlossFunction, normalProposal, data, [0, 0])

print("beta = ", betaGamma[0])
print("gamma = ", betaGamma[1])
print("Loss = ", loss)

input()