from sklearn.svm import SVC
import numpy as np
def partition(X, T, trainPart=0.7, shuffle=False,classification=False ):
    totalRows = len(X)
    nTrain = int(round(totalRows * trainPart))
    total = np.arange(totalRows)
    if shuffle:
        np.random.shuffle(total)
    trainIndices = total[:nTrain]
    testIndices = total[nTrain:]
    if classification:
        classes = np.unique(T)
        trainIndices = np.array([])
        testIndices = np.array([])
        for c in classes:
            rows =np.where(T == c)[0]
            n = round(trainPart * len(rows))
            trainIndices = np.hstack((trainIndices, rows[:n]))
            testIndices = np.hstack((testIndices, rows[n:]))
    testIndices = testIndices.astype(int)
    trainIndices=trainIndices.astype(int)

    Xtest=X[testIndices, :]
    Ttest=T[testIndices, :]
    Xtrain = X[trainIndices, :]
    Ttrain = T[trainIndices, :]

    return Xtrain, Ttrain, Xtest, Ttest

def percentCorrect(p, t):
    return np.sum(p.ravel()==t.ravel()) / float(len(t)) * 100

class SVCModel:
    model = SVC(kernel = 'rbf',gamma=2, C=1)
    def __init__(self, dataAsDataFrame, trainingPortion=0.7):       
        T = dataAsDataFrame.values[:,0:1]
        X = dataAsDataFrame.values[:,1:]
        Xtrain,Ttrain,Xtest,Ttest = partition(X,T,trainPart = 0.7, shuffle = True, classification = True)       

        #print("Training sample "+Ttrain[0]+"\t"+str(Ttrain.shape))
        #print("Training sample "+Xtrain[0]+"\t"+str(Xtrain.shape))
        print("Learning for classes"+np.unique(Ttest)+"\tNumber of samples: "+str(Ttest.shape[0]))
        #print("Testing sample "+Xtest[0]+"\t"+str(Xtest.shape))
        if len(np.unique(Ttest))>1:
            self.train(Xtrain,Ttrain)
            self.test(Xtest,Ttest)


    def train(self, Xtrain, Ttrain):
        print("Training classifier on "+str(len(Xtrain))+" samples")
        self.model.fit(Xtrain,Ttrain)
    
    def test(self, Xtest, Ttest):
        print("Testing classifier on "+(len(Xtest))+" samples")

        results = []
        results.append(self.model.predict(Xtest))    
        print(percentCorrect(results[0].reshape(-1,1),Ttest))

        
   