import numpy as np
import sys

def readData(pos,neg):
    pos_data=np.loadtxt(pos,skiprows=1)
    neg_data=np.loadtxt(neg,skiprows=1)
    data=dict()
    data['nPos']=len(pos_data)
    data['nNeg']=len(neg_data)
    data['n']=data['nPos']+data['nNeg']
    data['x']=np.vstack((pos_data,neg_data))
    data['y']=np.hstack((np.ones(data['nPos']),-np.ones(data['nNeg'])))
    return data

def arr2str(arr):
    s=''
    if len(arr.shape)==1:

        return ' '.join(map(str,arr))
    else:
        for subarr in arr:
            s+=arr2str(subarr)+'\n'
    return s

def RBF(a,b,sigma):
    return np.exp(-((a-b)**2).sum(1)/2.0/sigma**2)

class DualPerceptron:
    def __init__(self,train,test):
        self.train=train
        self.test=test
        
    def fit(self, kernel, sigma):
        # build gram matrix
        self.kernel=kernel
        self.sigma=sigma
        self.alpha=np.zeros(self.train['n'])
        gram_mat=np.zeros([self.train['n']]*2)
        homo=np.hstack((self.train['x'],np.ones([len(self.train['x']),1])))
        for i in range(self.train['n']):
            gram_mat[i]=kernel(homo[i], homo,sigma)
        #print gram_mat
        
        converge=False
        while converge==False:
            converge=True
            for i in range(train['n']):
                if self.train['y'][i]*sum(self.train['y']*self.alpha*gram_mat[i]) <= 0:
                    self.alpha[i]+=1
                    converge=False
    
    def predict(self):
        self.guess=np.zeros(self.test['n'])
        homo=np.hstack((self.train['x'],np.ones([len(self.train['x']),1])))
        homotest=np.hstack((self.test['x'],np.ones([len(self.test['x']),1])))
        for i,j in zip(homotest,range(self.test['n'])):
            if sum(self.train['y']*self.alpha*self.kernel(i,homo, self.sigma))<=0:
                self.guess[j]=-1
            else:
                self.guess[j]=1
                
    def evaluation(self):
        compare=self.guess-self.test['y']
        self.fp=sum(compare==2)
        self.fn=sum(compare==-2)
        self.err_rate=(self.fp+self.fn)/float(self.test['n'])
    
    def show(self):
        print "Alphas:", arr2str(self.alpha.astype(int))
        print "False positives:", self.fp
        print "False negatives:", self.fn
        print "Error rate:",self.err_rate*100.0,"%"

raw_path=[]
sigma=float(sys.argv[1])
for i in range(2,4+2):
    raw_path.append(sys.argv[i])
train=readData(raw_path[0],raw_path[1])
test=readData(raw_path[2],raw_path[3])
model=DualPerceptron(train,test)
model.fit(RBF,sigma)
model.predict()
model.evaluation()
model.show()
