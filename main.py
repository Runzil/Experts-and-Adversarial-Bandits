#ZAFEIRAKIS KONSTANTINOS 2019030035
import csv
import numpy as np
import math
import matplotlib.pyplot as plt


Reader = lambda: np.genfromtxt('milano_timeseries.csv', delimiter=',') #create a anonymous function to read the .csv file and return its components as a matrix[server][round]

def Regret_Finder(matrix, k , T): #function to find the index of the server with the least total load
    sums = np.zeros((k,))
    for server in range(k):
         sums[server] = np.sum(matrix[server])
    return np.argmin(sums)

def find_index(value, arr): #function to find the index of a given number in a given array
    for i in range(len(arr)):
        if arr[i] == value:
            return i
    return None

def experts(matrix , k , T):
    eta=np.sqrt(math.log(k)/T) #calculate eta variable
    W = np.ones((k, )) # weights array
    P = np.zeros((k,)) # probability array

    loss = np.zeros((k, T))  #loss / regret
    CURegret = np.zeros((T,))  #cummulative loss/regret

    ThComp = np.zeros((T,)) # Theoretical complexity

    RegretServer = Regret_Finder(matrix, k, T) #The server according to whom we will calculate the regret which is the server with the least overall load

    for round in range(T):                  #for every round
        for server in range(k):             #for every server
            P[server] = W[server] / sum(W)

        optimal = min(matrix.T[:][round]) #take the minimum value of all the servers at every round (minimum value of column round)

        pick = np.random.choice(range(len(P)), p=P) #pick the server based on the P probability

        for server in range(k):    #for every server
            loss[server][round] = matrix[server][round] - optimal      #calculating the loss of each server as the value of the server we picked minus the best server this time instant
            W[server] = W[server] * np.power((1-eta),loss[server][round])   #calculating the weights of each server


        # for  cummulative regret
        if round > 0:
            CURegret[round] = CURegret[round - 1] + matrix[pick][round] - matrix[RegretServer][round]  # vector keeping track of cummulative reward at all times
        else:
            CURegret[round] = matrix[pick][round] - matrix[RegretServer][round]

        ThComp[round] = 2 * (round * math.log(k))**(1/2) #calculating the theoretical complexity

    return CURegret , ThComp

def bandits(matrix , k , T):
    eta=pow((k * math.log(k)) / T , 1 / 3)#calculate eta variable
    epsilon = pow((k * math.log(k)) / T , 1 / 3)
    W = np.ones((k, )) # weights array
    P = np.zeros((k,)) # probability array
    Q = np.zeros((k,)) # probability array

    loss = np.zeros((k, ))  #loss
    CURegret = np.zeros((T,))  #cummulative loss/regret

    ThComp = np.zeros((T,)) # Theoretical complexity

    RegretServer = Regret_Finder(matrix, k, T) #The server according to whom we will calculate the regret which is the server with the least overall load

    for round in range(T):                  #for every round
        for server in range(k):             #for every server
            P[server] = W[server] / sum(W)
            Q[server] = (1-epsilon)*P[server] + epsilon/k

        optimal = min(matrix.T[:][round]) #take the minimum value of all the servers at every round (minimum value of column round)

        pick = np.random.choice(range(len(Q)), p=Q) # picking the server we select for this round according to Q

        loss[pick] = (matrix[pick][round] - optimal) / Q[pick] #calculating the loss of the picked server as the value of the server we picked minus the best server this time instant and we divide by Q of the picked server


        #for server in range(k):
        W[pick] = W[pick] * np.power((1-eta),loss[pick])

        # for cummulative regret
        if round > 0:
            CURegret[round] = CURegret[round - 1] + matrix[pick][round] - matrix[RegretServer][round]  # vector keeping track of cummulative reward at all times
        else:
            CURegret[round] = matrix[pick][round] - matrix[RegretServer][round]

        ThComp[round] = (k * round * math.log(k))**(1/2)    #calculating the theoretical complexity

    return CURegret , ThComp

def UCB(matrix , k , T):
    CURegret = np.zeros((T,))  #cummulative loss/regret
    RegretServer = Regret_Finder(matrix, k, T) #The server according to whom we will calculate the regret

    UCBRegret = np.zeros((T,))
    UCBComplexity = np.zeros((T,)) #theoretical complexity

    MU = np.zeros((k,))
    Q = np.zeros((k,))
    CULOAD = np.zeros((k,)) #cummulative load for each server
    UCB = [math.inf]*k

    for round in range(T):      #for every round

        pick = np.argmax(UCB)
        CULOAD[pick] = CULOAD[pick] - matrix[pick][round]
        Q[pick] =  Q[pick] + 1  # increase the number of times we selected a server

        MU[pick] = CULOAD[pick]/Q[pick]
        UCB[pick]= MU[pick] + math.sqrt(math.log(T) /  Q[pick])


        # for cummulative regret
        if round > 0:
            CURegret[round] = CURegret[round - 1] + matrix[pick][round] - matrix[RegretServer][round] # vector keeping track of cummulative reward at all times
        else:
            CURegret[round] = matrix[pick][round] - matrix[RegretServer][round]

        UCBComplexity[round] = math.sqrt(k * (round + 1) * math.log10(round + 1))
    return CURegret , UCBComplexity






#------------------------------------PLOTTERS---------------------------------------
def plotter(type1,type2,T,a,b):          #function for plotting has an optional second arguement in case we want to plot 2 functions in one grid
        fig, axs = plt.subplots(nrows=1, ncols=1)
        axs.set_title("Cummulative Regret for T= "+ str(T))
        axs.set_xlabel("Round T")
        axs.set_ylabel("Regret")
        axs.plot(np.arange(1, T + 1), a, label=type1)
        axs.plot(np.arange(1, T + 1), b, label=type2)
        axs.legend()

def bigplot(a1,a2,b1,b2,c,t1,t2,t3,t4,t5,T):
    fig, axs = plt.subplots(nrows=1, ncols=1)
    axs.set_title("Cummulative Regret for T= " + str(T))
    axs.set_xlabel("Round T")
    axs.set_ylabel("Regret")
    axs.plot(np.arange(1, T + 1), a1, label=t1)
    axs.plot(np.arange(1, T + 1), a2, label=t2)
    axs.plot(np.arange(1, T + 1), b1, label=t3)
    axs.plot(np.arange(1, T + 1), b2, label=t4)
    axs.plot(np.arange(1, T + 1), c, label=t5)
    axs.legend()

#------------------------------------TESTING---------------------------------------
T=7000
k=30
data=Reader() #read the .csv
EXP , EXPTH  = experts(data,k,T)
BAN , BANTH = bandits(data,k,T)
UCBp , UCBpol = UCB(data,k,T)

plotter("Expert","bandit",T,EXP,BAN)
plotter("experts","experts complexity",T,EXP,EXPTH)
plotter("bandits","bandit complexity",T,BAN,BANTH)

bigplot(EXP,EXPTH,BAN,BANTH,UCBp,"experts","experts complexity","bandits","bandit complexity","UCB",T)

T=1000
EXP , EXPTH  = experts(data,k,T)
BAN , BANTH = bandits(data,k,T)
UCBp , UCBpol = UCB(data,k,T)

plotter("Expert","bandit",T,EXP,BAN)
plotter("experts","experts complexity",T,EXP,EXPTH)
plotter("bandits","bandit complexity",T,BAN,BANTH)

bigplot(EXP,EXPTH,BAN,BANTH,UCBp,"experts","experts complexity","bandits","bandit complexity","UCB",T)







plt.show()
