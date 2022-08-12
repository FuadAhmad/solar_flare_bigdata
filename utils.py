import numpy as np
import pandas as pd
import torch #as th
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

#Load data
def load(file_name):
    with open(file_name, 'rb') as fp:
        obj = pickle.load(fp)
    return obj

#extract X,Y from binary formated data
def get_XY_np_array(xy):
  '''extract X,Y from binary formated data'''
  X = []
  Y = []
  xy_len = xy.shape[0]
  for i in range(0, xy_len):
    yy = xy[i][1].decode('UTF-8') #float(xy[i][1].decode('UTF-8'))
    Y.append(yy)
    
    xx = xy[i][0] #(9,144)
    xx_len = xx.shape[0]
    tx = []
    for j in range(0, xx_len):
      row = np.zeros(len(xx[j]))
      for k in range(len(xx[j])):
        row[k] = xx[j][k]
      #print("len(xx[j]): ", len(xx[j]),  " xx[j]: ",xx[j])
      #print("len(row): ", len(row), " row: ",row)
      tx.append(row)
    X.append(np.array(tx))
  return np.array(X), Y #np.array(Y)


#-------------------Label conversion ---------------------------------------
#Binary classification --> label conversion to BINARY class
def get_binary_labels_from(labels_str):
    tdf = pd.DataFrame(labels_str, columns = ['labels'])
    data_classes= [0, 1, 2, 3]
    d = dict(zip(data_classes, [0, 0, 1, 1])) 
    arr = tdf['labels'].map(d, na_action='ignore')
    return arr.to_numpy()

def get_int_labels_from_str(labels_str):
    tdf = pd.DataFrame(labels_str, columns = ['labels'])
    data_classes= np.unique(tdf)# = ["cat", "dog", "mouse"]
    len_labels = data_classes.shape[0]
    d = dict(zip(data_classes, range(0, len_labels)))
    arr = tdf['labels'].map(d, na_action='ignore')
    return arr


#-------------------data transform 3D->2D->3D ------------------------------
#Takes 3D array(x,y,z) >> transpose(y,z) >> return (x,z,y)
def GetTransposed2D(arrayFrom):
    toReturn = []
    alen = arrayFrom.shape[0]
    for i in range(0, alen):
        toReturn.append(arrayFrom[i].T)
    
    return np.array(toReturn)

#Takes 3D array(x,y,z) >> Flatten() >> return (x*y,z)
def Make2D(array3D):
    toReturn = []
    x = array3D.shape[0]
    y = array3D.shape[1]
    for i in range(0, x):
        for j in range(0, y):
            toReturn.append(array3D[i,j])
    
    return np.array(toReturn)

#Transform instance(92400, 33) into(1540x60x33)
def Get3D_MVTS_from2D(array2D, windowSize):
    arrlen = array2D.shape[0]
    mvts = []
    for i in range(0, arrlen, windowSize):
        mvts.append(array2D[i:i+windowSize])

    return np.array(mvts)




#-------------------data Scaler ------------------------------------------
from sklearn.preprocessing import StandardScaler

#LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized

def GetStandardScaler(data2d):
    scaler = StandardScaler()
    scaler = scaler.fit(data2d)
    return scaler

def GetStandardScaledData(data2d):
    scaler = StandardScaler()
    scaler = scaler.fit(data2d)
    #print(scaler.mean_)
    data_scaled = scaler.transform(data2d)
    return data_scaled

def transform_scale_data(data3d, scaler):
    print("original data shape:", data3d.shape) 
    trans = GetTransposed2D(data3d)
    print("transposed data shape:", trans.shape)    #(x, 60, 33)
    data2d = Make2D(trans)
    print("2d data shape:", data2d.shape)    
    #  scaler = GetStandardScaler(data2d)
    data_scaled = scaler.transform(data2d)
    mvts_scalled = Get3D_MVTS_from2D(data_scaled, data3d.shape[2])#,60)
    print("mvts data shape:", mvts_scalled.shape)
    transBack = GetTransposed2D(mvts_scalled)
    print("transBack data shape:", transBack.shape)
    return transBack




#------------------- Graph data operations ------------------------------------------
def GetGraphAdjMtrx(squareMtx, thresolds, keep_weights=False): #Apply Thresolds to squareMtx
    graphs = []
    mtxLen = squareMtx.shape[0]
    for thr in thresolds:
        m = np.zeros((mtxLen,mtxLen))#r = []        
        for i in range(0,mtxLen):
            for j in range(0,mtxLen):
                if i == j:# or squareMtx[i,j] > thr:
                    m[i,j] = 1
                elif squareMtx[i,j] > thr:
                  if keep_weights == True:
                    m[i,j] = squareMtx[i,j]
                  else:
                    m[i,j] = 1
        graphs.append(m)#np.array(r))  
    return graphs[0]


def get_adj_mat(c, th=0, keep_weights=True):
  #print("Creating graph with th: ", th)
  n = c.shape[0]
  a = np.zeros((n,n))
  for i in range(n):
    for j in range(n):
      #print("before:", c[i,j])
      if(c[i,j]>th):
        if(keep_weights):
          a[i,j] = c[i,j]
          a[j,i] = c[j,i]
        else:
          a[i,j] = 1
          a[j,i] = 1
      #print("after:", a[i,j])
  return a

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


#data crawler method------------------------------------------------
def get_adjs_nats(X_3d, num_temporal_split = 4, th = 0):

    num_X = X_3d.shape[0]
    num_nodes = X_3d.shape[1] #25
    len_st = int(X_3d.shape[2] / num_temporal_split) #15
    
    #populating adjacency matrices and node attributes of train events
    adjs = np.zeros((num_X, num_temporal_split, num_nodes, num_nodes))
    nats = np.zeros((num_X, num_temporal_split, num_nodes, len_st))
    
    for i in range(num_X):
      #print('Event: ', i)
      mt = X_3d[i].T #X_train[i].T[:,0:25] #consider first 25 solar params
      #mt = normalize_node_attributes(mt) ++++++++++++++++++++++++++++++
      for j in range(num_temporal_split):
        #print('Temporal split: ', j*15, (j+1)*15)
        smt = mt[j*len_st:(j+1)*len_st, : ]# len_st  mt[j*15:(j+1)*15,:]
        c_smt = np.corrcoef(smt.T)
        c_smt[np.isnan(c_smt)]=0
        #for l in range(0, num_nodes-1): #gcnconv will automatically add self loops
        #    c_smt[l,l] = 0
        #smt = normalize_node_attributes(smt)
        nats[i,j,:,:] = smt.T
        adj = GetGraphAdjMtrx(c_smt, [th], True)#get_adj_mat(c_smt, th, True) #change from ex 10

        adjs[i,j,:,:]=adj
    
    return adjs, nats


def build_edge_index_tensor(adj):
  num_nodes = adj.shape[0]
  source_nodes_ids, target_nodes_ids = [], []
  for i in range(num_nodes):
    for j in range(num_nodes):
      if(adj[i,j]==1):
        source_nodes_ids.append(i)
        target_nodes_ids.append(j)
  edge_index = np.row_stack((source_nodes_ids, target_nodes_ids))
  edge_index_tensor = torch.from_numpy(edge_index)
  return edge_index_tensor

def get_edge_index_weight_tensor(adj):
  num_nodes = adj.shape[0]
  source_nodes_ids, target_nodes_ids, edge_weights = [], [], []
  for i in range(num_nodes):
    for j in range(num_nodes):
      if(adj[i,j]>0):
        source_nodes_ids.append(i)
        target_nodes_ids.append(j)
        edge_weights.append(adj[i,j])
  edge_index = np.row_stack((source_nodes_ids, target_nodes_ids))
  edge_index_tensor = torch.from_numpy(edge_index)
  edge_weights_np = np.asarray(edge_weights, dtype=np.float32)
  edge_weights_tensor = torch.from_numpy(edge_weights_np)
  #print("Index shape: ",edge_index_tensor.shape)
  #print("Weight shape: ",edge_weights_tensor.shape)
  #print(edge_index_tensor)
  #print(edge_weights_tensor)
  return edge_index_tensor, edge_weights_tensor

def normalize_node_attributes(mvts):
  sc = StandardScaler()
  mvts_std = sc.fit_transform(mvts)
  return mvts_std





#-------------------Run Training epochs, get_results ------------------------------------------
# RunEpochs get_accuracy trian test acc
def RunEpochs(num_epochs = 1, print_loss_interval = 5): 
  for epoch in range(num_epochs):
    for i in range(num_train):#num_train
      model.zero_grad()

      class_scores = model(train_adjs[i], train_nats[i]) 
      #target = [y_train[i]]
      target = torch.from_numpy(np.array([y_train[i]]))
      target = target.to(device)
      loss = loss_function(class_scores, target)
      loss.backward()
      optimizer.step()
    if(epoch % print_loss_interval == 0):
      print ("epoch n loss:", epoch, loss)

#------------------------------train acc
def get_train_accuracy():
  num_train = X_train.shape[0]
  with torch.no_grad():
    numCorrect = 0
    for i in range(num_train):
      train_class_scores = model(train_adjs[i], train_nats[i])
      class_prediction = torch.argmax(train_class_scores, dim=-1) 
  
      if(class_prediction == y_train[i]): 
        numCorrect = numCorrect + 1
    return numCorrect/num_train


#---------test acc
def get_test_accuracy():
  num_test = X_test.shape[0]
  with torch.no_grad():
    numCorrect = 0
    for i in range(num_test):
      test_class_scores = model(test_adjs[i], test_nats[i]) #(adj_mat_array, node_att_array)
      class_prediction = torch.argmax(test_class_scores, dim=-1) 
      
      if(class_prediction == y_test[i]): 
        numCorrect = numCorrect + 1
    return numCorrect/num_test

def get_accuracy():
  print ("train_accuracy:", get_train_accuracy())
  print ("test_accuracy: ", get_test_accuracy())

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix, adjusted_rand_score

def get_accuracy_report_by_running_epochs(epochs, epoch_interval):
  maxAcc=0
  max_classification_report_dict=0
  max_acc_epoch = 0
  num_test = X_test.shape[0]

  for epoch in range(epoch_interval, epochs, epoch_interval):
    print("current epoch: ", epoch)
    RunEpochs(num_epochs = epoch_interval, print_loss_interval = 300)
    
    #get_accuracy()
    with torch.no_grad():
      numCorrect = 0
      predictaedLabel=[]
      for i in range(num_test):
        test_class_scores = model(test_adjs[i], test_nats[i]) #(adj_mat_array, node_att_array)
        class_prediction = torch.argmax(test_class_scores, dim=-1) 
        predictaedLabel.append(class_prediction)
        if(class_prediction == y_test[i]): 
          numCorrect = numCorrect + 1
      acc = numCorrect/num_test
      if acc  > maxAcc: #fgdg=round(acc, 2)
        maxAcc=acc
        max_acc_epoch = epoch
        max_classification_report_dict=metrics.classification_report(y_test, predictaedLabel, digits=3,output_dict=True)

  return maxAcc, max_acc_epoch, max_classification_report_dict




def doClassSpecificCalulcation(Accuracy,trainLebel,classification_report_dict):
  print('\np.mean(Accuracy) :',np.mean(Accuracy))
  print('\np.std(Accuracy) :',np.std(Accuracy))
  print('\n33333333 p.mean np.std(Accuracy) :     ',np.round(np.mean(Accuracy),2),"+-",np.round(np.std(Accuracy),2) )
  for j in [0, 1, 2, 3]:#np.unique(trainLebel ): #. len(...) np.unique(trainLebel):  [0 1 2 3]
    print('\n\n\n\nclass :',j) 
    precision=[]
    recall=[]
    f1_score=[]
    for i in range(len(classification_report_dict)):
      report=classification_report_dict[i]
      #print('classification_report : \n',report) 
      temp=report[str(j)]['precision'] 
      precision.append(temp)

      temp=report[str(j)]['recall'] 
      recall.append(temp)

      temp=report[str(j)]['f1-score'] 
      f1_score.append(temp)

    print('\np.mean(precision) \t p.mean(recall) \t p.mean(f1_score) :') 


    print(np.mean(precision)) 
    print(np.mean(recall)) 
    print(np.mean(f1_score))

    print('\np.mean p.std(precision) \tp.mean  p.std(recall) \tp.mean  p.std(f1_score) :')

    print(np.round(np.mean(precision),2),"+-",np.round(np.std(precision),2) )
    print(np.round(np.mean(recall),2),"+-",np.round(np.std(recall),2) )
    print(np.round(np.mean(f1_score),2),"+-",np.round(np.std(f1_score),2) )
