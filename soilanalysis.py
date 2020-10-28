########################## FUNCTIONS ###########################################
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression as LR
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.ensemble import IsolationForest
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
# import google.colab.drive as gd
import pdb
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten#, Conv1D

from scipy.signal import savgol_filter as sgf
from scipy.signal import detrend
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d
# import google.colab.drive as gd

"""======================== LWR FUNCTION ===================================="""
def lwr(d,y,x,tau): # Locally Weighted Regression
#   pdb.set_trace()  
  d=d.T  
  d = np.r_[np.ones((1,d.shape[1])),d]  # Incepction term
  x = np.append(1,x)                      # Incepction term
  n,m = d.shape # n: data dimenstion, m: number of training samples
  w = np.array([np.exp(-np.sum((x-d[:,i])**2)/(tau)) for i in range(m)])
  w = np.c_[w]
  Xj = np.array([np.diag(d[i,:]) for i in range(n)])
  A = np.array([w.T@Xj[i,:,:]@d.T for i in range(n)])
  A = np.squeeze(A)
  B = np.array([np.dot(np.dot(w.T,Xj[i,:,:]),y)  for i in range(n)])
  if (B<1e-6).all():
    theta=np.zeros(len(x))
    pdb.set_trace()
  else:
#     if np.linalg.det(A)==0:
#         pdb.set_trace()
    theta = np.linalg.inv(A)@B
  return np.squeeze(theta.T@x)

"""======================= GET_PCA FUNCTION ================================ """
def get_pca(num_principles,data,normalize=False):
  """ This function gives the principle components of the data.
  num_principles: number of principles
  data: an m x n matrix. n: data dimension, m: number of samples
  normalize: should the data be normalized or not """
  pca = PCA(n_components=num_principles)
  if normalize:  
    data = StandardScaler().fit_transform(data) # Normalize data (mean zero and unit variance)
  return pca.fit_transform(data)

"""======================= LWR_PCR FUNCTION ================================="""
def LW_pcr(num_principles,data,ref,n_cv,tau,test_size=[],normalize=False):#,test_index=[]):  # Locally weighted PCR
  """ This function predicts the output based on LWR model.
  num_principles: number of principles intended for data
  data: an m x n matrix with m and n being the number of samples and the dimension, respectively.
  ref: an m x ns matrix including reference values. ns indicates the number of sensors.
  n_cv: number of cross validations
  test_size: the size of test set """
  if test_size==[]:
    test_size = int(data.shape[0]/2)
  test_index = np.zeros((test_size,n_cv))
  R2 = np.zeros((n_cv,ref.shape[1]))    
  ld,ns = ref.shape  # ld: number of data samples, ns: number of sensors
  lwr_data = np.empty((n_cv,test_size,ref.shape[1]))  
  for k in range(n_cv):
    ri = random.sample(range(ld),test_size)  # random index
    test_index[:,k]=ri
    data_test, data_train = data[ri,:], np.delete(data,ri,axis=0)
    ref_test, ref_train = ref[ri,:], np.delete(ref,ri,axis=0)
    data_pca_train = get_pca(num_principles,data_train,normalize)
    data_pca_test = get_pca(num_principles,data_test,normalize)
    for i in range(ns):
      temp = np.array([lwr(data_pca_train,ref_train[:,i],data_pca_test[j,:],tau[i]) for j in range(test_size)])
      lwr_data[k,:,i] = np.squeeze(temp)
    R2[k,:] = np.array([r2_score(ref_test[:,ii],lwr_data[k,:,ii]) for ii in range(ns)])
  max_index = np.argmax(R2,axis=0)  # Which test set works the best for each parameter?
  lwr_data_best = np.empty((test_size,ns))
  for i in range(ns):
    lwr_data_best[:,i] = lwr_data[max_index[i],:,i]
  return lwr_data_best, np.max(R2,axis=0), np.array([test_index[:,i] for i in max_index])

"""======================= LWR_PCR1 FUNCTION ================================"""
def LW_pcr1(num_principles,data,ref,test_index,tau,normalize=False,scoring='val'):  # Locally weighted PCR for just one sensor type
  test_size=len(test_index)
  # nd, ns = test_size, ref.shape[1]  # nd: number of data samples, ns: number of sensors
  data_test, data_train = data[test_index,:], np.delete(data,test_index,axis=0)
  ref_test, ref_train = ref[test_index], np.delete(ref,test_index,axis=0)
  data_pca_train = get_pca(num_principles,data_train,normalize)
  data_pca_test = get_pca(num_principles,data_test,normalize)
  test_size=len(ref_test)
  # Prediction
#   pdb.set_trace()
  if scoring=='val':
      lwr_data = np.array([lwr(data_pca_train,ref_train,data_pca_test[j,:],tau) for j in range (test_size)])
  # lr_data = np.squeeze(lr_data)
  # R2 = r2_score(ref_test,lr_data)
      MSE = np.sum((lwr_data-ref_test)**2)/(len(lwr_data)-1)#mean_squared_error(ref_test,lr_data)  
      SD=np.sum((ref_test-np.mean(ref_test))**2)/(len(ref_test)-1)
      lccc=comp_lccc(lwr_data,ref_test)
  else:
#       pdb.set_trace()  
      lwr_data = np.array([lwr(data_pca_train,ref_train,data_pca_train[j,:],tau) for j in range(len(ref_train))])
  # lr_data = np.squeeze(lr_data)
  # R2 = r2_score(ref_test,lr_data)
      MSE = np.sum((lwr_data-ref_train)**2)/(len(lwr_data)-1)#mean_squared_error(ref_test,lr_data)  
      SD=np.sum((ref_train-np.mean(ref_train))**2)/(len(ref_train)-1)
      lccc=comp_lccc(lwr_data,ref_train)    
  R2=1-MSE/SD
  rpd=1/np.sqrt(1-R2)
  rmse=np.sqrt(MSE)
  # NRMSE = rmse/(np.max(ref_test)-np.min(ref_test))
  return lwr_data, R2, rpd, rmse, lccc

"""===================== comp_lccc FUNCTION ======================================"""
def comp_lccc(x,y):
  ''' COMPUTES Lin’s Concordance Correlation AS DESCRIBED HERE: https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/PASS/Lins_Concordance_Correlation_Coefficient.pdf
  X: THE VECTOR OF PREDICTIONS
  Y: REFERENCE VALUES'''

  sig_x,sig_y=np.std(x),np.std(y)
  omega=sig_x/sig_y
  nu = np.abs(np.mean(x)-np.mean(y))/np.sqrt(sig_x*sig_y)
  xi=2/(nu**2+omega+1/omega)
  return np.corrcoef(x,y)[0,1]*xi

"""============================ PLSR FUNCTION ==============================="""
def pls(num_principles,data,ref,test_index,scoring='val',n_cv=4):  # PLSR prediction model for just one sensor type
  data_test, data_train = data[test_index,:], np.delete(data,test_index,axis=0)
  ref_test, ref_train = ref[test_index], np.delete(ref,test_index)
  model = PLSRegression(num_principles)
  model.fit(data_train,ref_train)
  if scoring=='val':
    cal_data = model.predict(data_train).squeeze()  # Result of calibration set
    val_data = model.predict(data_test).squeeze()   # Result of validation set
    noise=val_data-ref_test
    MSE = np.sum((val_data-ref_test)**2)/(len(val_data)-1)#mean_squared_error(ref_test,lr_data)
    SD=np.sum((ref_test-np.mean(ref_test))**2)/(len(ref_test)-1)
    IQ=np.percentile(ref_test,75)-np.percentile(ref_test,25)
    lccc=comp_lccc(val_data,ref_test)
  else:
    plsr_data = cross_val_predict(model,data_train,ref_train,cv=n_cv).squeeze()
    MSE = np.sum((plsr_data-ref_train)**2)/(len(plsr_data)-1)#mean_squared_error(ref_test,lr_data)
    SD=np.sum((ref_train-np.mean(ref_train))**2)/(len(ref_train)-1)
    IQ=np.percentile(ref_train,75)-np.percentile(ref_train,25)
    lccc=comp_lccc(plsr_data,ref_train)
    
  R2=1-MSE/SD
  rpd=1/np.sqrt(1-R2)
  rmse=np.sqrt(MSE)
  rpiq=IQ/rmse
  # NRMSE = rmse/(np.max(ref_test)-np.min(ref_test))
  if scoring=='val':
      return val_data, cal_data, noise, rpiq, rpd, rmse, lccc,model
  else:
      return plsr_data, rpiq, rpd, rmse, lccc,model

"""================ Outlier detection with random forest =================== """
def outlier_removal(num_principles,data,ref,test_index):
  train_index=np.delete(np.arange(len(ref)),test_index)
  data_test, data_train = data[test_index,:], data[train_index,:]
  ref_test, ref_train = ref[test_index], ref[train_index]

  data_pca_train = get_pca(num_principles,data_train,normalize=True)
  data_pca_test = get_pca(num_principles,data_test,normalize=True)
  od = IsolationForest()
  od.fit(data_pca_train.T)
  y=od.predict(data_pca_train.T)
  data_train=np.delete(data_train,y==-1,axis=0)
  ref_train=np.delete(ref_train,y==-1)
  train_index=np.delete(train_index,y==-1)
  y=od.predict(data_pca_test.T)
  data_test=np.delete(data_test,y==-1,axis=0)
  ref_test=np.delete(ref_test,y==-1)
  test_index=np.delete(test_index,y==-1)
  return data_train,data_test,ref_train,ref_test,train_index,test_index

'''====================== Spectrum Fusion Function =========================='''
def spectrum_fusion(num_principles,spectrum1,spectrum2,ref,test_index,nc1=2,nc2=2,use_principle_components=False):
  ''' spectrum1 and spectrum2: m x n matrices. m: number of samples, n: dimension of each spectrum'''
  if use_principle_components:        
    PC1,PC2 = get_pca(nc1,spectrum1),get_pca(nc2,spectrum2) # result: num_principles x m matrices
    fused_spectrum=np.c_[PC1,PC2]
  else:
    fused_spectrum=np.hstack((spectrum1,spectrum2))
  return pls(num_principles,fused_spectrum,ref,test_index)


"""======================== PAF_CNN FUNCTION =============================== """
# def paf_cnn(spec1,spec2,nc,ref,n_cv,test_index,epok=200,normalize=False):
#   c1=get_pca(nc,spec1,normalize)  # output size: nc x m
#   c2=get_pca(nc,spec2,normalize)  # output size: nc x m
#   # c3=get_pca(nc,spec3.T,normalize)  # output size: nc x m
#   c1,c2=np.expand_dims(c1.T,axis=2),np.expand_dims(c2.T,axis=2)#,np.expand_dims(c3.T,axis=2)
#   im=np.concatenate((c1,c2),axis=2)  # output: images with size nc x m x 3    
#   m=len(ref)
#   # *******************  BUILD CNN *********************
#   model=build_cnn(nc)
#   #******************************************************
#   test_size = len(test_index)
#   cnn_data = np.empty((n_cv,test_size))  
#   R2,MSE = np.zeros(n_cv),np.zeros(n_cv)
#   # test_index = np.zeros((test_size,n_cv))
#   for k in range(n_cv):
#     if k==0:
#       ri=test_index
#     else:
#       ri = random.sample(range(m),test_size)  # random index
#     # test_index[:,k]=ri
#     im_test, im_trn = im[ri,:,:], np.delete(im,ri,axis=0)
#     y_test, y_trn = ref[ri], np.delete(ref,ri)
#     im_test, im_trn = im_test.reshape(test_size,nc,2,1), im_trn.reshape(m-test_size,nc,2,1)
#     model.fit(im_trn, y_trn, epochs=epok, batch_size=3, verbose=0)
#     cnn_data[k,:]=np.array([np.squeeze(model.predict(im_test[j,:,:,0].reshape(1,nc,2,1))) for j in range(test_size)])
#     MSE[k] = np.sum((cnn_data[k,:].squeeze()-y_test)**2)/(len(cnn_data[k,:])-1)#mean_squared_error(y_test,cnn_data)
#     SD=np.sum((y_test-np.mean(y_test))**2)/(len(y_test)-1)
#     R2[k]=1-MSE[k]/SD
    
#   mi = np.argmax(R2)  # Which test set works the best for each parameter?
#   r2=R2[mi]
#   rpd=1/np.sqrt(1-r2)
#   rmse=np.sqrt(MSE[mi])
#   nrmse=rmse/(np.max(y_test)-np.min(y_test))
#   cnn_data_best = np.empty((test_size,ns))
#   return cnn_data_best, r2,rpd,rmse,nrmse  #np.array([test_index[:,i] for i in max_index])

"""======================== PAF_CNN2 FUNCTION =============================== """
def paf_cnn2(spec1,spec2,nc,ref,test_index,epok=1100,normalize=False):    
#   pdb.set_trace()      
#   m=len(ref)
  c1=get_pca(nc,spec1,normalize)  # output size: m x nc
  c2=get_pca(nc,spec2,normalize)  # output size: m x nc
#   c1+=np.abs(np.min(c1)) # Setting min to 0
#   c1=(c1*255/np.max(c1)).astype(int)   # Setting max to 255
#   c2+=np.abs(np.min(c2)) # Setting min to 0
#   c2=(c2*255/np.max(c2)).astype(int)   # Setting max to 255  
  c1,c2=np.expand_dims(c1,axis=2),np.expand_dims(c2,axis=2)
  im=np.concatenate((c1,c2),axis=2)  # output: images with size m x nc x 2    
#   pdb.set_trace()    
  # *******************  BUILD CNN *********************
  model = build_cnn(nc)
  #******************************************************
  train_index=np.delete(np.arange(len(ref)),test_index)
    
  im_test, im_trn = im[test_index,:,:], im[train_index,:,:]
  y_test, y_trn = ref[test_index], ref[train_index]
  # im_test, im_trn = im[test_index,:,:], np.delete(im,test_index,axis=0)
  im_test, im_trn = im_test.reshape(len(test_index),nc,2,1), im_trn.reshape(len(train_index),nc,2,1)
  model.fit(im_trn, y_trn, epochs=epok, batch_size=3, verbose=0)
  # history=model.fit(im_trn, y_trn, epochs=epok, batch_size=3, verbose=0)
#   pdb.set_trace()
  # plt.plot(history.history['loss'])
  # plt.plot(history.history['val_acc'])
  # plt.title('model accuracy')
  # plt.ylabel('Loss')
  # plt.xlabel('epoch')
  # plt.legend(['train', 'test'], loc='upper left')
  # plt.show()
  # pdb.set_trace()
#   cnn_data=np.array([np.squeeze(model.predict(im_test[j,:,:,0].reshape(1,nc,2,1))) for j in range(len(test_index))])
  cnn_data=np.array([np.squeeze(model.predict(j.reshape(1,nc,2,1))) for j in im_test])
  # R2 = r2_score(y_test,cnn_data)
  MSE = np.sum((cnn_data.squeeze()-y_test)**2)/(len(cnn_data)-1)#mean_squared_error(y_test,cnn_data)
  SD=np.sum((y_test-np.mean(y_test))**2)/(len(y_test)-1)
  R2=1-MSE/SD
  rmse=np.sqrt(MSE)
  # NRMSE = rmse/(np.max(y_test)-np.min(y_test))
  return cnn_data.squeeze(), R2, 1/np.sqrt(1-R2), rmse, comp_lccc(cnn_data.squeeze(),y_test)

"""======================== PAF_CNN3 FUNCTION =============================== """
# def paf_cnn3(spec1,spec2,spec3,spec4,nc,ref,test_index,epok=1100,normalize=False):  
#   m=len(ref)
#   c1=get_pca(nc,spec1,normalize)  # output size: nc x m
#   c2=get_pca(nc,spec2,normalize)  # output size: nc x m
#   c3=get_pca(nc,spec3,normalize)  # output size: nc x m
#   c4=get_pca(nc,spec4,normalize)  # output size: nc x m
#   c1,c2,c3,c4=np.expand_dims(c1,axis=2),np.expand_dims(c2,axis=2),np.expand_dims(c3.T,axis=2),np.expand_dims(c4.T,axis=2)
#   im=np.concatenate((c1,c2,c3,c4),axis=2)  # output: images with size nc x m x 3    
#   # *******************  BUILD CNN *********************
#   model = build_cnn2(nc)
#   #******************************************************
#   train_index=np.delete(np.arange(len(ref)),test_index)
#   im_test, im_trn = im[test_index,:,:], im[train_index,:,:]
#   y_test, y_trn = ref[test_index], ref[train_index]
#   # im_test, im_trn = im[test_index,:,:], np.delete(im,test_index,axis=0)
#   im_test, im_trn = im_test.reshape(len(test_index),nc,4,1), im_trn.reshape(len(train_index),nc,4,1)
#   # model.fit(im_trn, y_trn, epochs=epok, batch_size=3, verbose=0)
#   history=model.fit(im_trn, y_trn, epochs=epok, batch_size=3, verbose=0)
#   # pdb.set_trace()
#   plt.plot(history.history['acc'])
#   # plt.plot(history.history['val_acc'])
#   # plt.title('model accuracy')
#   plt.ylabel('acc')
#   plt.xlabel('epoch')
#   # plt.legend(['train', 'test'], loc='upper left')
#   plt.show()
#   # pdb.set_trace()
#   cnn_data=np.array([np.squeeze(model.predict(im_test[j,:,:,0].reshape(1,nc,4,1))) for j in range(len(test_index))])
#   # R2 = r2_score(y_test,cnn_data)
#   MSE = np.sum((cnn_data.squeeze()-y_test)**2)/(len(cnn_data)-1)#mean_squared_error(y_test,cnn_data)
#   SD=np.sum((y_test-np.mean(y_test))**2)/(len(y_test)-1)
#   R2=1-MSE/SD
#   rmse=np.sqrt(MSE)
#   NRMSE = rmse/(np.max(y_test)-np.min(y_test))
#   return cnn_data.squeeze(), R2, 1/np.sqrt(1-R2), rmse, NRMSE

"""======================== Build a CNN2 model FUNCTION ===================== """
# def build_cnn2(nc):
#   model=Sequential()
#   model.add(Conv2D(filters=3,kernel_size=2,activation='relu',input_shape=(nc,4,1)))
#   model.add(Conv2D(filters=3,kernel_size=3,activation='relu'))
#   model.add(Flatten())
#   # model.add(Dense(8,activation='relu'))
#   model.add(Dense(1,activation='relu'))
#   model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
#   return model

"""======================== Build a CNN model FUNCTION ===================== """
def build_cnn(nc):
  model=Sequential()
  y=model.add(Conv2D(filters=3,kernel_size=(2,2),activation='relu',input_shape=(nc,2,1)))
#   pdb.set_trace()  
#   model.add(Conv1D(filters=3,kernel_size=3,activation='relu',input_shape=(nc-1,1,3)))
  model.add(Flatten())
  # model.add(Dense(8,activation='relu'))
  model.add(Dense(1,activation='relu'))
  model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
  return model

"""======================= ICOV_EST FUNCTION ================================"""
def icov_est(data,ref,W,lr=1,er=1):
  """ This function estimates the inverse of covariance
  data: data to be fused. Each col. indicates a data. Number of rows indicates number of samples. ref: a vector
  er: detoriation from zero
  lr: learning rate
  W: initial W                                                            """
  m, n = data.shape # m: size of data, n: number of sensors
  # W = .5*np.random.rand(n,n)
  # W = W + W.T # to make C symmetric
  H = np.ones((n,1))
  J = np.array([1000])
  J_best = np.array([1000])
  # pdb.set_trace()
  W_best = W
  k=0
  while True:
    # if len(J)%50 == 0:
      # pdb.set_trace()
    t1, t2 = H.T @ W @ H, H.T @ W @ data.T
    x_hat = t2 / t1
    d = (x_hat-ref)
    J = np.append(J, .5 * np.sum(d**2) / m)
    if J[-1] < J_best:
      # pdb.set_trace()
      J_best = J[-1]
      W_best = W
    if J[-1] < er:
      break
    # pdb.set_trace()
    for i in range(n):
      # pdb.set_trace()
      W[i,i] = W[i,i] - lr/m * np.sum(d * (data[:,i]*t1 - t2))/t1**2
      for j in range(i+1,n):
        W[i,j] = W[i,j] - lr/m  * np.sum(d * ((data[:,i]+data[:,j])*t1 - 2*t2))/t1**2
        W[j,i] = W[i,j]
    lr *= 1.001
    k += 1
    if k>1000:
      break
  # pdb.set_trace()
  return W_best

"""======================= ICOV_EST_NN FUNCTION ============================="""
# def icov_est_nn(data,ref):
#   # This function estimates the inverse of covariance by using an ANN
#   # data: data to be fused. Each col. indicates a data. Number of rows indicates number of samples. ref: a vector
#   from keras.models import Sequential
#   from keras.layers import Dense

#   m, n = data.shape # m: size of data, n: number of sensors
#   model = Sequential()
#   model.add(Dense(n,Activation = 'linear',input_shpe=(n,)))
#   model.add(Dense(1,Activation = 'linear'))
#   model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
#   return W_best

"""===================== SENSORS_FUSION FUNCTION ============================"""
#def sensors_fusion(data,ref,test_size,ann=False):
#  """" data: data to be fused. Each col. indicates a data. Number of rows indicates number of samples. 
#  ref: a vector
#  test_size: number of samples used as test set """
#  ld, num_sensors = data.shape
#  n_cv = 2  # number of cross-validations
#  MSE = np.zeros(n_cv)
#  R2 = np.empty(n_cv)  
#  for k in range(n_cv):    
#    ri = random.sample(range(ld),test_size)  # random index
#    data_test, data_train = data[ri,:], np.delete(data,ri,axis=0)
#    ref_test, ref_train = ref[ri], np.delete(ref,ri)    
#
#    N = data_train.shape[0]
#    
#    # Build covariance matrix
#    C = np.zeros((num_sensors,num_sensors))
#    for i in range(num_sensors):
#      for j in range(num_sensors):
#        sig2 = np.sum((data_train[:,i]-ref_train)*(data_train[:,j]-ref_train))/(N-1)
#        C[i,j] = sig2
#
#    if ann:
#      C_inv = icov_est(data_train,ref_train,np.linalg.inv(C),lr=1)
#      # C_inv = icov_est_nn(data_train,ref_train)
#    else:
#      C_inv = np.linalg.inv(C)
#    P = 1 / np.sum(np.sum(C_inv,axis=1))
#    fused_data = P * np.dot(np.sum(C_inv,axis=0),data_test.T)
#    MSE[k] = np.sum((fused_data.squeeze()-ref_test)**2)/(len(fused_data)-1)#mean_squared_error(ref_test,fused_data) #
#    SD=np.sum((ref_test-np.mean(ref_test))**2)/(len(ref_test)-1)
#    R2[k]=1-MSE[k]/SD
#    R2_max = np.max(R2)
#    rmse = np.sqrt(MSE[np.argmax(R2)])
#    rpd = RPD[np.argmax(R2)]
#  return rmse, R2_max, rpd,comp_lccc()


def sensors_fusion(data,ref,test_index):
  """" data: data to be fused. Each col. indicates a data. Number of rows indicates number of samples. 
  ref: a vector
  test_size: number of samples used as test set """
  ld, num_sensors = data.shape
  # n_cv = 2  # number of cross-validations
#  MSE = np.zeros(n_cv)
#  R2 = np.empty(n_cv)  
  C = np.zeros((num_sensors,num_sensors))
#  for k in range(n_cv):
  ri=test_index
  data_test, data_train = data[ri,:], np.delete(data,ri,axis=0)
  ref_test, ref_train = ref[ri], np.delete(ref,ri)    
 
  N = data_train.shape[0]
  
  # Build covariance matrix
  
  for i in range(num_sensors):
    for j in range(num_sensors):
      sig2 = np.sum((data_train[:,i]-ref_train)*(data_train[:,j]-ref_train))/(N-1)
      C[i,j] = sig2
  
  C_inv = np.linalg.inv(C)
  P = 1 / np.sum(np.sum(C_inv,axis=1))
  fused_data = P * np.dot(np.sum(C_inv,axis=0),data_test.T)
  MSE = np.sum((fused_data.squeeze()-ref_test)**2)/(len(fused_data)-1)#mean_squared_error(ref_test,fused_data) #
  SD=np.sum((ref_test-np.mean(ref_test))**2)/(len(ref_test)-1)
  IQ=np.percentile(ref_test,75)-np.percentile(ref_test,25)
  r2=1-MSE/SD
  rmse = np.sqrt(MSE)
  rpd = 1/np.sqrt(1-r2)
  rpiq=IQ/rmse
  lccc=comp_lccc(fused_data.squeeze(),ref_test)
  return rpiq, rpd, rmse, lccc, C
  """=================== SENSOR FUSION 2 FUNCTION ==========================="""
def sensors_fusion2(data,ref,test_size,test_index=[], ann=False):
  """" data: data to be fused. Each col. indicates a data. Number of rows indicates number of samples.
  ref: a vector
  test_size: number of samples used as test set """
  ld, num_sensors = data.shape
  # pdb.set_trace()
  if ann:
    n_cv=1
  else:
    n_cv = 5  # number of cross-validations      
    test_index = np.zeros((test_size,n_cv))
  MSE = np.zeros(n_cv)
  R2 = np.zeros(n_cv)
  for k in range(n_cv):
    if ann:
      data_test, data_train = data[test_index.astype(int),:], np.delete(data,test_index.astype(int),axis=0)
      ref_test, ref_train = ref[test_index.astype(int)], np.delete(ref,test_index.astype(int))      
    else:
      ri = random.sample(range(ld),test_size)  # random index
      test_index[:,k]=ri
      data_test, data_train = data[ri,:], np.delete(data,ri,axis=0)
      ref_test, ref_train = ref[ri], np.delete(ref,ri)    

    N = data_train.shape[0]
    
    # Build covariance matrix
    C = np.zeros((num_sensors,num_sensors))
    for i in range(num_sensors):
      for j in range(num_sensors):
        sig2 = np.sum((data_train[:,i]-ref_train)*(data_train[:,j]-ref_train))/(N-1)
        C[i,j] = sig2

    if ann:
      C_inv = icov_est(data_train,ref_train,np.linalg.inv(C),lr=1)
      # C_inv = icov_est_nn(data_train,ref_train)
    else:
      C_inv = np.linalg.inv(C)    
    P = 1 / np.sum(np.sum(C_inv,axis=1))
    fused_data = P * np.dot(np.sum(C_inv,axis=0),data_test.T)
    # pdb.set_trace()
    MSE[k] = np.sum((fused_data.squeeze()-ref_test)**2)/(len(fused_data)-1)#mean_squared_error(ref_test,fused_data) #
    SD=np.sum((ref_test-np.mean(ref_test))**2)/(len(ref_test)-1)
    R2[k]=1-MSE[k]/SD
  if ann:    
    return np.min(MSE), np.max(R2)
  else:
    best_test_set = test_index[:,np.argmin(MSE)]
    return np.min(MSE), np.max(R2), best_test_set

"""======================== Fusion by linear regression ====================="""
def fuse_lr(data,ref,ri):
  ld, num_sensors = data.shape
#  n_cv = 1  # number of cross-validations
#  MSE = np.zeros(n_cv)
#  R2 = np.zeros(n_cv)  
#  for k in range(n_cv):    
#  ri = random.sample(range(ld),test_size)  # random index
  data_test, data_train = data[ri,:], np.delete(data,ri,axis=0)
  ref_test, ref_train = ref[ri], np.delete(ref,ri)
  model = LR().fit(data_train,ref_train)
  y = model.predict(data_test)
  MSE=np.sum((y-ref_test)**2)/(len(y)-1)#mean_squared_error(ref_test,y)
  SD=np.sum((ref_test-np.mean(ref_test))**2)/(len(ref_test)-1)
  IQ=np.percentile(ref_test,75)-np.percentile(ref_test,25)
  r2=1-MSE/SD
  rmse = np.sqrt(MSE)
  rpd = 1/np.sqrt(1-r2)
  rpiq=IQ/rmse
  lccc=comp_lccc(y,ref_test)
  
  return rpiq, rpd, rmse, lccc

'''=========== Continum removal pre-processing method function =============='''
def continuum_remove(wl,spec):
  n=len(spec)
  cr=np.empty(np.shape(spec))
  xx=np.linspace(wl[0],wl[-1],num=len(wl))
  for l in range(n):
    i = spec[l,:]
    ii = i.tolist()
    mat = np.column_stack((wl,ii))
    hull = ConvexHull(mat)
    x=mat[hull.simplices,0].ravel()
    y=mat[hull.simplices,1].ravel()
    yinterp = interp1d(x, y)
    cr[l,:]=spec[l,:]-yinterp(xx)
  return cr[:,1:]
'''====================== Moving average function ==========================='''
def moving_average(x,k,axis=1):
  ''' Returns the simple moving average of a matrix 
  x: input matrix
  k: the size of moving average window
  axis: the axis along with moving average is applied'''
  if axis==1:
    x=x.T
  # pdb.set_trace()
  n=x.shape[0]
  a=np.ones(n)
  a[k:]=0
  mam=np.array([np.roll(a,i) for i in range(n-k+1)])
  ma=1/k*np.dot(mam,x)
  if axis==1:
    return ma.T
  else:
    return ma

'''========================= Removing slope function ========================'''
def de_slope(x,mid_point):
  p=x.copy()
  for i in range(1,mid_point):
    if x[i]>p[i-1]:
      p[i]=p[i-1]
    else:
      p[i]=x[i]
  for i in range(len(x)-1,mid_point,-1):
    if x[i-1]>p[i]:
      p[i-1]=p[i]
    else:
      p[i-1]=x[i-1]
  
  return x-p

'''=============== Modify the noise of calibration predictions =============='''
def modify_data(val,cal,ref,ti):
  sig2_val=np.sum((val-ref[ti])**2)/(len(val)-1)
  sig2_cal=np.sum((cal-np.delete(ref,ti))**2)/(len(cal)-1)
  # pdb.set_trace()
  if sig2_val > sig2_cal:
    noise=np.random.normal(scale=np.sqrt(sig2_val-sig2_cal),size=len(cal))
    cal=cal+noise
  d=np.zeros(len(ref))
  d[ti],d[np.delete(np.arange(len(ref)),ti)]=val,cal
  return d

'''============ Variable selection for optimum PLS result =================='''
def varselpls(x,y,ti,nc,step=1):
  '''
  X: m x n data, m: number of samples
  Y: m x 1 reference values
  TI: test indices
  NC: number of principal components (latent variables)
  Returns the following outputs:
  IND_OPT: optimum indices (variables) of X
  RMSE_OPT: the RMSE that is reached by the optimal varible selection
  '''
  ci=np.arange(x.shape[0])# calibration index
  ci=np.delete(ci,ti)
  plsModel=PLSRegression(n_components=nc)
  plsModel.fit(x[ci,:],y[ci])
  reg_coe=np.abs(plsModel.coef_[:,0])
  a=np.sort(reg_coe)[:-nc:step]
  la=len(a)
  rmse_opt=np.zeros(la)
  for c,k in zip(a,range(la)):
    var_sel=reg_coe>=c
    x_cal=x[ci,:][:,var_sel]
    rmse_opt[k]=cross_val_score(plsModel,x_cal,y[ci],cv=4,scoring='neg_mean_squared_error').mean()
#  pdb.set_trace()
  k_opt=np.argmax(rmse_opt)
  ind_opt=np.arange(x.shape[1])[reg_coe>=a[k_opt]]
  rmse_opt=np.sqrt(-rmse_opt[k_opt])
  return ind_opt, rmse_opt

'''============= Correction of joint part of NIR spectrum ==================='''
def correct_nir_disjoint(nir_spec,nir_wl):
    interval=np.logical_and(nir_wl>1000,nir_wl<1050)
    y=nir_spec[:,interval]
    # ax=plt.subplots(1,1,figsize=(20,20))[1]
    # ax.plot(nir_wl,nir_spec[1,:])
    d=(y[:,-1]-y[:,0])/np.sum(interval)
    for i in range(1,np.sum(interval)):
      y[:,i]=y[:,i-1]+d
    # yy=np.array([x+d for x in y[:,1:].T])
    # y[:,1:]=yy.T
    nir_spec[:,interval]=y
    # ax.plot(nir_wl,nir_spec[1,:],'r')
    # plt.show()
    return nir_spec
  
''' ===================== Outlier detection =============================='''
def outlier_mahalonobis(nir_spec,spec2,ref,percent1=5,percent2=95):
    pc=get_pca(8,nir_spec)
    C=np.linalg.inv(np.cov(pc.T))
    mpc=np.mean(pc,axis=0)
    md_nir=np.array([np.sqrt(np.c_[x-mpc].T@C@np.c_[x-mpc]) for x in pc]).squeeze()
    T1,T2=np.percentile(md_nir,percent1),np.percentile(md_nir,percent2)
#     pdb.set_trace()
    di=np.logical_and(md_nir>=T1,md_nir<=T2) # Drop Indices
    nir_spec=nir_spec[di,:]
    spec2=spec2[di,:]
#     pdb.set_trace()
    ref.drop(ref[~di].index,inplace=True)
    return nir_spec,spec2,ref
######################## END OF FUNCTIONS SECTION ##############################