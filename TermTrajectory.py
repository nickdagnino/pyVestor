from importlib.resources import path
from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import numpy as np
#from sklearn.preprocessing import MinMaxScaler

class TermTrajectory:

    global file

    print("Start:")

   # file = open("TermTrajectory.txt", 'w')
    #file.write("TermTrajectory.txt\n")
    #file.write("Start:\n")
    #file.write("Pulling data: H:\My Drive\Vs Code\Code\stockMarketData-AAL.csv\n")

    global dS
    global dateX 
    global closeY
    global clusterSetX
    global clusterSetY
    global coefs

    dataSet = pd.read_csv('stockMarketData-AAL.csv') #H:\My Drive\Vs Code\Code\stockMarketData-TSLA.csv
    dS = dataSet.sort_values('Date')
    dS.head()

    closeY = []
    for i in dS['Close']:
        closeY.append(i)
    
    dateX = []
    for i in range(0, len(closeY)):
        dateX.append(i)

    clusterSetX = []
    clusterSetY = []

    coefs = []


    dS = dataSet.sort_values('Date')
    dS.head()    

    def clusterX(div):
        print("Creating %s clustered data sets for the x variable." %div)
        for i in range(0, len(dateX), div):
            clusterSetX.append(dateX[slice(i, i+div)])
        #file.write("%s clustered data sets for the x variable have been created.  Printed to file. \n" %div)
        #file.write('%s\n' %clusterSetX)

    def clusterY(div):
        print("Creating clustered data for the y variable.")
        for i in range(0, len(closeY), div):
            clusterSetY.append(closeY[slice(i, i+div)])
        #file.write("%s clustered data sets for the x variable have been created.  Printed to file. \n" %div)
        #file.write('%s\n' %clusterSetY)
    
    def findCoefs():
        print("Finding coefficents for each section.")
        for i in range(len(clusterSetX)):
            coefs.append(np.poly1d(np.polyfit(clusterSetX[i], clusterSetY[i], 1)))
        #file.write("All coefficents have been found.\n")
        #file.write('%s\n' %coefs)

    clusterX(50)
    clusterY(50)

    findCoefs()

    plt.figure(figsize = (18,9))
    plt.plot(range(dS.shape[0]),(dS['Close']))
    plt.xlabel('Date',fontsize=18)
    plt.ylabel('Close',fontsize=18)

    for i in range(len(coefs)):
        coef = np.polyfit(clusterSetX[i], clusterSetY[i], 1)
        poly1d_fn = np.poly1d(coef)
        plt.plot(clusterSetX[i], poly1d_fn(clusterSetX[i]),'-r')
    
    plt.show()
    
    print("End:")
    #file.write("End:")
   # file.close()
