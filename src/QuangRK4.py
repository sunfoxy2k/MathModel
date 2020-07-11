import matplotlib.pyplot as plt

# Input Values
BETA = float(input("Enter beta: "))
GAMMA = float(input("Enter gamma: "))
N = int(input("Enter the total number of people: "))
INIT_INFECTED = int(input("Enter number of initial infected cases: "))
INIT_RECOVERED = int(input("Enter number of initial recovered cases: "))
TIME = int(input("Enter the time to get the approximation: "))


# SIR Class
class SIRValue:
    def __init__(self, suspected_num, infected_num, recovered_num):
        self.suspected = suspected_num
        self.infected = infected_num
        self.recovered = recovered_num

    def showValue(self):
        suspected_str = str(self.suspected)
        infected_str = str(self.infected)
        recovered_str = str(self.recovered)

        print("Number of Suspected People: " + suspected_str)
        print("Number of Infected People: " + infected_str)
        print("Number of Recovered People: " + recovered_str)

    def getTotalPopulation(self):
        return self.suspected + self.infected + self.recovered


class SIRRatio:
    def __init__(self, infected_rate, recovered_rate):
        self.infected = infected_rate
        self.recovered = recovered_rate


class Slop:
    def __init__(self, slop_value):
        self.value = slop_value


class SIRRungeKuttaSlop:
    def __init__(self, slop_values):
        self.values = slop_values


# SIR Helper Functions
def calculateSuspected(N, sir_value):
    infected_num = sir_value.infected
    recovered_num = sir_value.recovered
    return N - infected_num - recovered_num


def printSIRValues(sir_values, dt):
    i = 0
    for value in sir_values:
        print("\nWeek " + str(i))
        print("Suspected: " + str(value.suspected))
        print("Infected: " + str(value.infected))
        print("Recovered: " + str(value.recovered))
        i += dt


# Derivative Functions
def dSdt(t, sir_value, sir_ratio):
    infected_num = sir_value.infected
    suspected_num = sir_value.suspected
    total_population = sir_value.getTotalPopulation()

    # Infected ratio per person
    infected_rate = sir_ratio.infected
    infected_ratio_per_person = infected_rate / total_population

    return -(infected_ratio_per_person * (suspected_num*infected_num))


def dIdt(t, sir_value, sir_ratio):
    #diffSuspected = dSdt(t, sir_value, sir_ratio)
    #diffRecovered = dRdt(t, sir_value, sir_ratio)
    #return -diffSuspected - diffRecovered
    return sir_value.infected * (sir_value.suspected*sir_ratio.infected/sir_value.getTotalPopulation() - sir_ratio.recovered)

def dRdt(t, sir_value, sir_ratio):
    infected_num = sir_value.infected
    recovered_rate = sir_ratio.recovered

    return (recovered_rate * infected_num)


# RungeKutta Helper Functions
def calculateInitialSlop(time, sir_value, sir_ratio):
    KS = dSdt(time, sir_value, sir_ratio)
    KI = dIdt(time, sir_value, sir_ratio)
    KR = dRdt(time, sir_value, sir_ratio)

    suspected_slop = Slop(KS)
    infected_slop = Slop(KI)
    recovered_slop = Slop(KR)

    K1Slop_value = [suspected_slop, infected_slop, recovered_slop]

    return SIRRungeKuttaSlop(K1Slop_value)


def calculateMiddleSlop(time, dt, sir_value, sir_ratio, prev_slop):
    suspected_num = sir_value.suspected
    infected_num = sir_value.infected
    recovered_num = sir_value.recovered

    prev_sus_slop = prev_slop[0]
    prev_inf_slop = prev_slop[1]

    k_suspected_num = suspected_num + (dt*prev_sus_slop.value / 2)
    k_infected_num = infected_num + (dt*prev_inf_slop.value / 2)
    k_sir_value = SIRValue(k_suspected_num, k_infected_num, recovered_num)

    KS = dSdt(time + dt/2, k_sir_value, sir_ratio)
    KI = dIdt(time + dt/2, k_sir_value, sir_ratio)
    KR = dRdt(time + dt/2, k_sir_value, sir_ratio)

    suspected_slop = Slop(KS)
    infected_slop = Slop(KI)
    recovered_slop = Slop(KR)

    KSlop_value = [suspected_slop, infected_slop, recovered_slop]

    return SIRRungeKuttaSlop(KSlop_value)


def calculateLastSlop(time, dt, sir_value, sir_ratio, prev_slop):
    suspected_num = sir_value.suspected
    infected_num = sir_value.infected
    recovered_num = sir_value.recovered

    prev_sus_slop = prev_slop[0]
    prev_inf_slop = prev_slop[1]

    k_suspected_num = suspected_num + (dt*prev_sus_slop.value)
    k_infected_num = infected_num + (dt*prev_inf_slop.value)
    k_sir_value = SIRValue(k_suspected_num, k_infected_num, recovered_num)

    KS = dSdt(time + dt, k_sir_value, sir_ratio)
    KI = dIdt(time + dt, k_sir_value, sir_ratio)
    KR = dRdt(time + dt, k_sir_value, sir_ratio)

    suspected_slop = Slop(KS)
    infected_slop = Slop(KI)
    recovered_slop = Slop(KR)

    KSlop_value = [suspected_slop, infected_slop, recovered_slop]

    return SIRRungeKuttaSlop(KSlop_value)


def calculateNextValue(curr_value, dt, K):
    K1 = K[0].value
    K2 = K[1].value
    K3 = K[2].value
    K4 = K[3].value
    difference = (dt/6) * (K1 + 2*K2 + 2*K3 + K4)

    return curr_value + difference


# RungeKutta SIR Model Function
def approximateRK4SIR(sir_value, sir_ratio, t, dt=1):
    suspected_num = sir_value.suspected
    infected_num = sir_value.infected
    recovered_num = sir_value.recovered

    sir_values = []
    curr_sir_value = SIRValue(suspected_num, infected_num, recovered_num)
    sir_values.append(curr_sir_value)

    prev_slop = [None]*3
    KS = [None]*4
    KI = [None]*4
    KR = [None]*4

    for i in range(0, t, dt):
        # K1
        slop = calculateInitialSlop(i, sir_values[i], sir_ratio)
        prev_slop = slop.values
        KS[0] = prev_slop[0]
        KI[0] = prev_slop[1]
        KR[0] = prev_slop[2]

        # K2
        slop = calculateMiddleSlop(i, dt, sir_values[i], sir_ratio, prev_slop)
        prev_slop = slop.values
        KS[1] = prev_slop[0]
        KI[1] = prev_slop[1]
        KR[1] = prev_slop[2]

        # K3
        slop = calculateMiddleSlop(i, dt, sir_values[i], sir_ratio, prev_slop)
        prev_slop = slop.values
        KS[2] = prev_slop[0]
        KI[2] = prev_slop[1]
        KR[2] = prev_slop[2]

        # K4
        slop = calculateLastSlop(i, dt, sir_values[i], sir_ratio, prev_slop)
        prev_slop = slop.values
        KS[3] = prev_slop[0]
        KI[3] = prev_slop[1]
        KR[3] = prev_slop[2]

        # Next SIR Value
        suspected_num = calculateNextValue(suspected_num, dt, KS)
        infected_num = calculateNextValue(infected_num, dt, KI)
        recovered_num = calculateNextValue(recovered_num, dt, KR)

        # Add to SIR Array
        curr_sir_value = SIRValue(suspected_num, infected_num, recovered_num)
        sir_values.append(curr_sir_value)

    return sir_values


def RK4SIR(total_population, I0, R0, beta, gamma, time, dt):
    sir_value = SIRValue(0, I0, R0)
    sir_value.suspected = calculateSuspected(total_population, sir_value)

    sir_ratio = SIRRatio(beta, gamma)

    approximated_values = approximateRK4SIR(sir_value, sir_ratio, time, dt)

    return approximated_values


# Testing SIR Model
sir_values = RK4SIR(N, INIT_INFECTED, INIT_RECOVERED, BETA, GAMMA, TIME, 1)
printSIRValues(sir_values, 1)

# Plot Graph
x = []
for i in range(0, TIME + 1):
    x.append(i)

Sus = []
Inf = []
Rec = []
for value in sir_values:
    Sus.append(value.suspected)
    Inf.append(value.infected)
    Rec.append(value.recovered)

plt.plot(x, Sus, label="Suspected")
plt.plot(x, Inf, label="Infected")
plt.plot(x, Rec, label="Recovered")

plt.xlabel('Week')
plt.ylabel('People')

plt.title('SIR-RK4 Approximation')

plt.legend()
plt.show()

input()