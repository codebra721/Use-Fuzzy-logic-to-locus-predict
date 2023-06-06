# -*- coding: utf-8 -*-

import struct
'''from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from std_msgs.msg import Float32MultiArray'''
import numpy as np
import pandas as pd
import cupy as cp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from scipy.stats import norm
import tensorflow as tf
from keras.models import Sequential, Model, load_model
from keras.layers import LSTM,Lambda, Dense, Activation, Input, RepeatVector, TimeDistributed, Bidirectional, GRU, Dropout, concatenate,Subtract,ELU
from keras.callbacks import ModelCheckpoint,EarlyStopping
import sys
np.set_printoptions(suppress=True)

rback5 = np.genfromtxt('C:/Users/liyuchen/SynologyDrive/ICCV_LAB stuff/billiards/20200531/rebound/backlabelrebound0531.csv',delimiter=',')
rleft5 = np.genfromtxt('C:/Users/liyuchen/SynologyDrive/ICCV_LAB stuff/billiards/20200531/rebound/leftlabelrebound0531.csv',delimiter=',')#40
rright5 = np.genfromtxt('C:/Users/liyuchen/SynologyDrive/ICCV_LAB stuff/billiards/20200531//rebound/rightlabelrebound0531.csv',delimiter=',')#33
rtop5 = np.genfromtxt('C:/Users/liyuchen/SynologyDrive/ICCV_LAB stuff/billiards/20200531/rebound/toplabelrebound0531.csv',delimiter=',')#18
rleftback5 = np.genfromtxt('C:/Users/liyuchen/SynologyDrive/ICCV_LAB stuff/billiards/20200531/rebound/leftbacklabelrebound0531.csv',delimiter=',')#47
rlefttop5 = np.genfromtxt('C:/Users/liyuchen/SynologyDrive/ICCV_LAB stuff/billiards/20200531/rebound/lefttoplabelrebound0531.csv',delimiter=',')#47
rrightback5 = np.genfromtxt('C:/Users/liyuchen/SynologyDrive/ICCV_LAB stuff/billiards/20200531/rebound/rightbacklabelrebound0531.csv',delimiter=',')#45
rrighttop5 = np.genfromtxt('C:/Users/liyuchen/SynologyDrive/ICCV_LAB stuff/billiards/20200531/rebound/righttoplabelrebound0531.csv',delimiter=',')#44

def rread(traject,n):#將反彈軌跡切分成500與100
  t500=traject[:500]
  t250=traject[500:600]
  t500[np.isnan(t500)]=0
  t250[np.isnan(t250)]=0
  ball5_500=np.zeros([500,n*4])
  ball5_500[:,:n*4]=t500
  return ball5_500,t250
ballc=12  #big
ballc2=12   #small

rw1=28#反彈最長軌跡數
rw2=51
rw3=44
rw4=32
rw5=28
rw6=40
rw7=24
rw8=38
rback5_500,rback5_100=rread(rback5,rw1)
rleft5_500,rleft5_100=rread(rleft5,rw2)
rright5_500,rright5_100=rread(rright5,rw3)
rtop5_500,rtop5_100=rread(rtop5,rw4)
rleftback5_500,rleftback5_100=rread(rleftback5,rw5)
rlefttop5_500,rlefttop5_100=rread(rlefttop5,rw6)
rrightback5_500,rrightback5_100=rread(rrightback5,rw7)
rrighttop5_500,rrighttop5_100=rread(rrighttop5,rw8)
print(rback5_500)

back5 = np.genfromtxt('C:/Users/liyuchen/SynologyDrive/ICCV_LAB stuff/billiards/20200531/label/backlabel0531.csv', delimiter=',')  # 48
left5 = np.genfromtxt('C:/Users/liyuchen/SynologyDrive/ICCV_LAB stuff/billiards/20200531/label/leftlabel0531.csv', delimiter=',')  # 56
right5 = np.genfromtxt('C:/Users/liyuchen/SynologyDrive/ICCV_LAB stuff/billiards/20200531/label/rightlabel0531.csv', delimiter=',')  # 49
top5 = np.genfromtxt('C:/Users/liyuchen/SynologyDrive/ICCV_LAB stuff/billiards/20200531/label/toplabel0531.csv', delimiter=',')  # 43
leftback5 = np.genfromtxt('C:/Users/liyuchen/SynologyDrive/ICCV_LAB stuff/billiards/20200531/label/leftbacklabel0531.csv',delimiter=',')#47
lefttop5 = np.genfromtxt('C:/Users/liyuchen/SynologyDrive/ICCV_LAB stuff/billiards/20200531/label/lefttoplabel0531.csv',delimiter=',')#47
rightback5 = np.genfromtxt('C:/Users/liyuchen/SynologyDrive/ICCV_LAB stuff/billiards/20200531/label/rightbacklabel0531.csv',delimiter=',')#45
righttop5 = np.genfromtxt('C:/Users/liyuchen/SynologyDrive/ICCV_LAB stuff/billiards/20200531/label/righttoplabel0531.csv',delimiter=',')#44

def read(traject,n):
  t500=traject[:500]
  t250=traject[500:600]
  t500[np.isnan(t500)]=0
  t250[np.isnan(t250)]=0
  ball5_500=np.zeros([500,n*4])
  ball5_500[:,:n*4]=t500
  return ball5_500,t250

w1=43#第一組最長軌跡
w2=47
w3=42
w4=39
w5=42
w6=41
w7=37
w8=40
back5_500,back5_100=read(back5,w1)
left5_500,left5_100=read(left5,w2)
right5_500,right5_100=read(right5,w3)
top5_500,top5_100=read(top5,w4)
leftback5_500,leftback5_100=read(leftback5,w5)
lefttop5_500,lefttop5_100=read(lefttop5,w6)
rightback5_500,rightback5_100=read(rightback5,w7)
righttop5_500,righttop5_100=read(righttop5,w8)
print(back5_500)
'''
def plot3d(top5_500,w1,a,top5_5002,w2):
            x2=np.zeros([w2])
            y2=np.zeros([w2])
            z2=np.zeros([w2])

            for i in range(w2):
                x2[i]=top5_5002[a,4*i+1]
                y2[i]=top5_5002[a,4*i+2]
                z2[i]=top5_5002[a,4*i+3]
            
            x=np.zeros([w1])
            y=np.zeros([w1])
            z=np.zeros([w1])

            for i in range(w1):
                x[i]=top5_500[a,4*i+1]
                y[i]=top5_500[a,4*i+2]
                z[i]=top5_500[a,4*i+3]
                #print(back5_500[0,3*i+1])
            for i in range(w1-1):
                if y[i+1] > y[i]:
                    break

            fig = plt.figure(figsize=(8,6))
            ax = fig.add_subplot(111, projection='3d')

            ax.scatter(x[:i],y[:i],z[:i],c = 'b', marker='o')
            ax.scatter(x[i:],y[i:],z[i:],c = 'r', marker='o')
            #ax.scatter(x2,y2,z2,c = 'c', marker='o')
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            ax.set_zlabel('Z-axis')

            

            print(math.atan((x[i]-x[i-1])/(y[i-1]-y[i]))*180/math.pi)
            print(math.atan((z[i]-z[i-1])/(y[i-1]-y[i]))*180/math.pi)

            print(math.atan((x[i+1]-x[i])/(y[i+1]-y[i]))*180/math.pi)
            print(math.atan((z[i+1]-z[i])/(y[i+1]-y[i]))*180/math.pi)
            print('\n')
            j=i
            i=j-1
            print(math.atan((x[i]-x[i-1])/(y[i-1]-y[i]))*180/math.pi)
            print(math.atan((z[i]-z[i-1])/(y[i-1]-y[i]))*180/math.pi)
            i=j+1
            print(math.atan((x[i+1]-x[i])/(y[i+1]-y[i]))*180/math.pi)
            print(math.atan((z[i+1]-z[i])/(y[i+1]-y[i]))*180/math.pi)
            #i=j
            print('\n')
            #j=i
            i=j-2
            print(math.atan((x[i]-x[i-1])/(y[i-1]-y[i]))*180/math.pi)
            print(math.atan((z[i]-z[i-1])/(y[i-1]-y[i]))*180/math.pi)
            i=j+2
            print(math.atan((x[i+1]-x[i])/(y[i+1]-y[i]))*180/math.pi)
            print(math.atan((z[i+1]-z[i])/(y[i+1]-y[i]))*180/math.pi)
            #i=j

            print('\n')
            #j=i
            i=j-3

            print(math.atan((x[i]-x[i-1])/(y[i-1]-y[i]))*180/math.pi)
            print(math.atan((z[i]-z[i-1])/(y[i-1]-y[i]))*180/math.pi)
            i=j+3
            print(math.atan((x[i+1]-x[i])/(y[i+1]-y[i]))*180/math.pi)
            print(math.atan((z[i+1]-z[i])/(y[i+1]-y[i]))*180/math.pi)


            print('\n')
            i=j
            i=j-2
            print(math.atan((x[i+1]-x[i-1])/(y[i-1]-y[i+1]))*180/math.pi)
            print(math.atan((z[i+1]-z[i-1])/(y[i-1]-y[i+1]))*180/math.pi)
            i=j+2
            print(math.atan((x[i+1]-x[i-1])/(y[i+1]-y[i-1]))*180/math.pi)
            print(math.atan((z[i+1]-z[i-1])/(y[i+1]-y[i-1]))*180/math.pi)
            plt.show()
            return x,y,z,j

m=410
x,y,z,j=plot3d(rright5_500,rw3,440,right5_500,w3)
x,y,z,j=plot3d(rleft5_500,rw2,126,left5_500,w2)
x,y,z,j=plot3d(rtop5_500,rw4,m,top5_500,w4)
x,y,z,j=plot3d(rback5_500,rw1,m,back5_500,w1)       
x,y,z,j=plot3d(rlefttop5_500,rw6,421,lefttop5_500,w6)

def mix(tra1,tra2,w1,w2,a):
            t1=np.zeros([w1])
            x1=np.zeros([w1])
            y1=np.zeros([w1])
            z1=np.zeros([w1])
            t2=np.zeros([w2])
            x2=np.zeros([w2])
            y2=np.zeros([w2])
            z2=np.zeros([w2])
            for i in range(w1):
                t1[i]=tra1[a,4*i]
                x1[i]=tra1[a,4*i+1]
                y1[i]=tra1[a,4*i+2]
                z1[i]=tra1[a,4*i+3]
            for i in range(w2):
                t2[i]=tra2[a,4*i]
                x2[i]=tra2[a,4*i+1]
                y2[i]=tra2[a,4*i+2]
                z2[i]=tra2[a,4*i+3]
            for i in range(w2):
                if t2[i]==t1[0]:
                    print(i)
                    break
            for j in range(w2-i):
                if t2[i+j]!=t1[j]:
                    print(i+j-1)
                    break
            print(y1)
            print(y2)

            t3=np.zeros([w1+i])
            x3=np.zeros([w1+i])
            y3=np.zeros([w1+i])
            z3=np.zeros([w1+i])
            t4=np.zeros([w1+i])
            x4=np.zeros([w1+i])
            y4=np.zeros([w1+i])
            z4=np.zeros([w1+i])
            t3[:i+j]=t2[:i+j]
            x3[:i+j]=x2[:i+j]
            y3[:i+j]=y2[:i+j]
            z3[:i+j]=z2[:i+j]
            t3[i+j:]=t1[j:]
            x3[i+j:]=x1[j:]
            y3[i+j:]=y1[j:]
            z3[i+j:]=z1[j:]

            t4[:i]=t2[:i]
            x4[:i]=x2[:i]
            y4[:i]=y2[:i]
            z4[:i]=z2[:i]
            t4[i:]=t1
            x4[i:]=x1
            y4[i:]=y1
            z4[i:]=z1

            print(y3)
            print(y4)
            for k in range(w1+i):
                if t3[k]==0:
                    break
            fig = plt.figure(figsize=(8,6))
            ax = fig.gca(projection='3d')
            ax.scatter(x3[:i+j+2],y3[:i+j+2],z3[:i+j+2],c = 'r', marker='o')
            ax.scatter(x3[i+j+2:k-10],y3[i+j+2:k-10],z3[i+j+2:k-10],c = 'b', marker='s')
            #ax.scatter(x3[2:i+j],y3[2:i+j],z3[2:i+j],c = 'b', marker='o')
            #ax.scatter(x4[i:k],y4[i:k],z4[i:k],c = 'r', marker='s')
            ##ax.scatter(x3,y3,z3,c = 'b', marker='o')
            #ax.scatter(x4,y4,z4,c = 'r', marker='o')
            #ax.scatter(x4[:k-10],y4[:k-10],z4[:k-10],c = 'r', marker='o')
            ax.set_xlabel('X-axis (cm)')
            ax.set_ylabel('Y-axis (cm)')
            ax.set_zlabel('Z-axis (cm)')

            plt.show()
            print(x3-x4)
            print(y3-y4)
            print(z3-z4)
mix(rlefttop5_500,lefttop5_500,rw6,w6,421)
'''
def drop(top5_500,w1,sample):#計算入射角與反射角
            x=np.zeros([sample,w1])
            y=np.zeros([sample,w1])
            z=np.zeros([sample,w1])
            angle=np.zeros([sample,2])
            for a in range(sample):
                for i in range(w1):
                    x[a,i]=top5_500[a,4*i+1]
                    y[a,i]=top5_500[a,4*i+2]
                    z[a,i]=top5_500[a,4*i+3]
                for i in range(w1-1):
                    if y[a,i+1] > y[a,i]:
                        break
                for k in range(w1-i):
                    if y[a,i+k] == 0 :
                        break
                #print(k)        
                if k>3:
                    j=i
                    i=j-2
                    x1=(math.atan((x[a,i+1]-x[a,i-1])/(y[a,i-1]-y[a,i+1]))*180/math.pi)
                    z1=(math.atan((z[a,i+1]-z[a,i-1])/(y[a,i-1]-y[a,i+1]))*180/math.pi)
                    i=j+2
                    x2=(math.atan((x[a,i+1]-x[a,i-1])/(y[a,i+1]-y[a,i-1]))*180/math.pi)
                    z2=(math.atan((z[a,i+1]-z[a,i-1])/(y[a,i+1]-y[a,i-1]))*180/math.pi)
                else:
                    j=i
                    i=j-1
                    x1=(math.atan((x[a,i]-x[a,i-1])/(y[a,i-1]-y[a,i]))*180/math.pi)
                    z1=(math.atan((z[a,i]-z[a,i-1])/(y[a,i-1]-y[a,i]))*180/math.pi)
                    i=j+1
                    x2=(math.atan((x[a,i+1]-x[a,i])/(y[a,i+1]-y[a,i]))*180/math.pi)
                    z2=(math.atan((z[a,i+1]-z[a,i])/(y[a,i+1]-y[a,i]))*180/math.pi)
                angle[a,0]=-x2
                angle[a,1]=z2
                #angle[a,0]=x1-x2
                #angle[a,1]=z2-z1
            return angle

angle1=drop(rback5_500,rw1,500)
angle2=drop(rleft5_500,rw2,500)
angle3=drop(rright5_500,rw3,500)
angle4=drop(rtop5_500,rw4,500)
angle5=drop(rleftback5_500,rw5,500)
angle6=drop(rlefttop5_500,rw6,500)
angle7=drop(rrightback5_500,rw7,500)
angle8=drop(rrighttop5_500,rw8,500)
bangle1=drop(rback5_100,rw1,100)
bangle2=drop(rleft5_100,rw2,100)
bangle3=drop(rright5_100,rw3,100)
bangle4=drop(rtop5_100,rw4,100)
bangle5=drop(rleftback5_100,rw5,100)
bangle6=drop(rlefttop5_100,rw6,100)
bangle7=drop(rrightback5_100,rw7,100)
bangle8=drop(rrighttop5_100,rw8,100)
#print(angle1)
        #for i in range(500):
         #   if angle2[i,1]>40:
          #      angle2[i]=angle2[i-1]
plt.plot(angle1[:,0],angle1[:,1],'>', label='down_spin',color='deepskyblue')
plt.plot(angle2[:,0],angle2[:,1],'p', label='left_spin',color='salmon')
plt.plot(angle3[:,0],angle3[:,1],'^', label='right_spin',color='orangered')
plt.plot(angle4[:,0],angle4[:,1],'*', label='up_spin',color='darkorange')
plt.plot(angle5[:,0],angle5[:,1],'v', label='lowerleft_spin',color='gold')
plt.plot(angle6[:,0],angle6[:,1],'o', label='upperleft_spin',color='purple')
plt.plot(angle7[:,0],angle7[:,1],'<', label='lowerright_spin',color='greenyellow')
plt.plot(angle8[:,0],angle8[:,1],'s', label='upperright_spin',color='dodgerblue')
plt.title('Degree of rotation')
plt.xlabel('Δθv')
plt.ylabel('Δθh')
#plt.legend()
#plt.show()
#print(angle4[m])

def trainc(tra500):#取前11個桌球軌跡當訓練集
            a=np.zeros([(ballc-ballc2+1)*500,ballc,3])
            train=np.zeros([500,ballc,3])
            for i in range(ballc):
                for j in range(500):
                    x=[]
                    y=[]
                    z=[]
                    x.append(tra500[j,1+4*i])
                    y.append(tra500[j,2+4*i])
                    z.append(tra500[j,3+4*i])
                    x=np.array(x)    
                    y=np.array(y)
                    z=np.array(z)
                    train[j,i,0]=x
                    train[j,i,1]=y
                    train[j,i,2]=z
                    if i > (ballc2-2):
                        a[500*(i-ballc2+1):500+500*(i-ballc2+1)]=train
            return a
a1c=trainc(back5_500)
a2c=trainc(left5_500)
a3c=trainc(right5_500)
a4c=trainc(top5_500)
a5c=trainc(leftback5_500)
a6c=trainc(lefttop5_500)
a7c=trainc(rightback5_500)
a8c=trainc(righttop5_500)
print(a1c)

def testc(tra100):
            b=np.zeros([(ballc-ballc2+1)*100,ballc,3])
            testn=np.zeros([100,ballc,3])
            for i in range(ballc):
                for j in range(100):
                    x=[]
                    y=[]
                    z=[]
                    x.append(tra100[j,1+4*i])
                    y.append(tra100[j,2+4*i])
                    z.append(tra100[j,3+4*i])
                    x=np.array(x)    
                    y=np.array(y)
                    z=np.array(z)
                    testn[j,i,0]=x
                    testn[j,i,1]=y
                    testn[j,i,2]=z
                if i > (ballc2-2):
                    b[100*(i-ballc2+1):100+100*(i-ballc2+1)]=testn
            return b
b1c=testc(back5_100)
b2c=testc(left5_100)
b3c=testc(right5_100)
b4c=testc(top5_100)
b5c=testc(leftback5_100)
b6c=testc(lefttop5_100)
b7c=testc(rightback5_100)
b8c=testc(righttop5_100)
print(b1c)

def slash(a1c,sample,angle1):#將資料整理成水平方向反彈角度差，垂直方向反彈角度差，1到12刻斜率
            s=np.zeros([sample,3])           
            s[:,1:]=angle1
            for i in range(sample):
                #print((a1c[i,0,0]-a1c[i,ballc-1,0])/(a1c[i,0,1]-a1c[i,ballc-1,1])*100)
                s[i,0]=(a1c[i,0,0]-a1c[i,ballc-1,0])/(a1c[i,0,1]-a1c[i,ballc-1,1])*100
                #s[i,1]=(a1c[i,0,2]-a1c[i,ballc-1,2])/(a1c[i,0,1]-a1c[i,ballc-1,1])*100
            return s
s1=slash(a1c,500,angle1)
s2=slash(a2c,500,angle2)
s3=slash(a3c,500,angle3)
s4=slash(a4c,500,angle4)
s5=slash(a5c,500,angle5)
s6=slash(a6c,500,angle6)
s7=slash(a7c,500,angle7)
s8=slash(a8c,500,angle8)
print(s8)

ntrain_x=np.zeros([(ballc-ballc2+1)*500*8,ballc,3])#reshapeX為桌球座標
i=0
ntrain_x[(ballc-ballc2+1)*500*i:(ballc-ballc2+1)*500*(i+1)]=a1c
i=i+1
ntrain_x[(ballc-ballc2+1)*500*i:(ballc-ballc2+1)*500*(i+1)]=a2c
i=i+1
ntrain_x[(ballc-ballc2+1)*500*i:(ballc-ballc2+1)*500*(i+1)]=a3c
i=i+1
ntrain_x[(ballc-ballc2+1)*500*i:(ballc-ballc2+1)*500*(i+1)]=a4c
i=i+1
ntrain_x[(ballc-ballc2+1)*500*i:(ballc-ballc2+1)*500*(i+1)]=a5c
i=i+1
ntrain_x[(ballc-ballc2+1)*500*i:(ballc-ballc2+1)*500*(i+1)]=a6c
i=i+1
ntrain_x[(ballc-ballc2+1)*500*i:(ballc-ballc2+1)*500*(i+1)]=a7c
i=i+1
ntrain_x[(ballc-ballc2+1)*500*i:(ballc-ballc2+1)*500*(i+1)]=a8c
print(ntrain_x.shape)

ntrain_y=np.zeros([(ballc-ballc2+1)*500*8,2])#reshapeY為入射角反射角
i=0
for j in range(ballc-ballc2+1):
  ntrain_y[500*j:500*j+500]=angle1
i=i+1
for j in range(ballc-ballc2+1):
  ntrain_y[(ballc-ballc2+1)*500*i+500*j:(ballc-ballc2+1)*500*i+500*j+500]=angle2
i=i+1
for j in range(ballc-ballc2+1):
  ntrain_y[(ballc-ballc2+1)*500*i+500*j:(ballc-ballc2+1)*500*i+500*j+500]=angle3
i=i+1
for j in range(ballc-ballc2+1):
  ntrain_y[(ballc-ballc2+1)*500*i+500*j:(ballc-ballc2+1)*500*i+500*j+500]=angle4
i=i+1
for j in range(ballc-ballc2+1):
  ntrain_y[(ballc-ballc2+1)*500*i+500*j:(ballc-ballc2+1)*500*i+500*j+500]=angle5
i=i+1
for j in range(ballc-ballc2+1):
  ntrain_y[(ballc-ballc2+1)*500*i+500*j:(ballc-ballc2+1)*500*i+500*j+500]=angle6
i=i+1
for j in range(ballc-ballc2+1):
  ntrain_y[(ballc-ballc2+1)*500*i+500*j:(ballc-ballc2+1)*500*i+500*j+500]=angle7
i=i+1
for j in range(ballc-ballc2+1):
  ntrain_y[(ballc-ballc2+1)*500*i+500*j:(ballc-ballc2+1)*500*i+500*j+500]=angle8
print(ntrain_y.shape)
'''
from keras.utils.vis_utils import plot_model
model = Sequential()
#model.add(LSTM(200, input_shape=(ballc,3,),activation='relu'))
model.add(Dense(200, input_shape=(ballc*3,),activation='relu'))
Dropout(0.5)
model.add(Dense(100,activation='relu'))
Dropout(0.5)
#model.add(Dense(50,activation='relu'))
#Dropout(0.5)
model.add(Dense(2))
model.compile(loss='mse',optimizer='adam',metrics=['acc'])
checkpoint = ModelCheckpoint('C:/Users/liyuchen/Desktop/504/tabletennies_model/classified0614_1.h5', monitor='acc', verbose=1, save_best_only=True,mode='max')
early_stopping = EarlyStopping(monitor='loss', patience=50, verbose=1, mode='min')
callbacks_list = [checkpoint,early_stopping]
'''
#plot_model(model, to_file='/content/drive/MyDrive/classified/model.png',show_shapes = True)
#from keras.utils.vis_utils import plot_model
model=load_model('C:/Users/liyuchen/Desktop/504/tabletennies_model/classified0614_1.h5')#23_8  11_2
#plot_model(model, to_file='/content/drive/MyDrive/classified/model_new.png',show_shapes = True)
'''
history = model.fit(ntrain_x.reshape(4000,ballc*3),ntrain_y,batch_size=100 ,epochs=5000,verbose=1,callbacks=callbacks_list)
model.save('C:/Users/liyuchen/Desktop/504/tabletennies_model/classified0614_1.h5')

plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

plt.plot(history.history['acc'])
plt.title('Model acc')
plt.ylabel('acc')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()
'''
plt.figure(figsize=(8,6))
plt.scatter(bangle1[:,0],bangle1[:,1],color='tomato',alpha=0.3,label="predict")
plt.scatter(bangle2[:,0],bangle2[:,1],color='tomato',alpha=0.3)
plt.scatter(bangle3[:,0],bangle3[:,1],color='tomato',alpha=0.3)
plt.scatter(bangle4[:,0],bangle4[:,1],color='tomato',alpha=0.3)
plt.scatter(bangle5[:,0],bangle5[:,1],color='tomato',alpha=0.3)
plt.scatter(bangle6[:,0],bangle6[:,1],color='tomato',alpha=0.3)
plt.scatter(bangle7[:,0],bangle7[:,1],color='tomato',alpha=0.3)
plt.scatter(bangle8[:,0],bangle8[:,1],color='tomato',alpha=0.3)

pred1=model.predict(b1c.reshape((ballc-ballc2+1)*100,ballc*3))#輸入軌跡，輸出為反彈角度差
pred2=model.predict(b2c.reshape((ballc-ballc2+1)*100,ballc*3))
pred3=model.predict(b3c.reshape((ballc-ballc2+1)*100,ballc*3))
pred4=model.predict(b4c.reshape((ballc-ballc2+1)*100,ballc*3))
pred5=model.predict(b5c.reshape((ballc-ballc2+1)*100,ballc*3))
pred6=model.predict(b6c.reshape((ballc-ballc2+1)*100,ballc*3))            
pred7=model.predict(b7c.reshape((ballc-ballc2+1)*100,ballc*3))
pred8=model.predict(b8c.reshape((ballc-ballc2+1)*100,ballc*3))

plt.scatter(pred1[:,0],pred1[:,1],color='cornflowerblue',alpha=0.25,marker='s',label="true")
plt.scatter(pred2[:,0],pred2[:,1],color='cornflowerblue',alpha=0.25,marker='s')
plt.scatter(pred3[:,0],pred3[:,1],color='cornflowerblue',alpha=0.25,marker='s')
plt.scatter(pred4[:,0],pred4[:,1],color='cornflowerblue',alpha=0.25,marker='s')
plt.scatter(pred5[:,0],pred5[:,1],color='cornflowerblue',alpha=0.25,marker='s')
plt.scatter(pred6[:,0],pred6[:,1],color='cornflowerblue',alpha=0.25,marker='s')
plt.scatter(pred7[:,0],pred7[:,1],color='cornflowerblue',alpha=0.25,marker='s')
plt.scatter(pred8[:,0],pred8[:,1],color='cornflowerblue',alpha=0.25,marker='s')
plt.legend(loc="upper right")

'''bs1=slash(b1c,100,pred1)#將資料整理成水平方向反彈角度差，垂直方向反彈角度差，1到12刻斜率
bs2=slash(b2c,100,pred2)
bs3=slash(b3c,100,pred3)
bs4=slash(b4c,100,pred4)
bs5=slash(b5c,100,pred5)
bs6=slash(b6c,100,pred6)
bs7=slash(b7c,100,pred7)
bs8=slash(b8c,100,pred8)'''
bs1=slash(b1c,100,bangle1)
bs2=slash(b2c,100,bangle2)
bs3=slash(b3c,100,bangle3)
bs4=slash(b4c,100,bangle4)
bs5=slash(b5c,100,bangle5)
bs6=slash(b6c,100,bangle6)
bs7=slash(b7c,100,bangle7)
bs8=slash(b8c,100,bangle8)
'''pred5001=model.predict(a1c.reshape((ballc-ballc2+1)*500,ballc*3))
pred5002=model.predict(a2c.reshape((ballc-ballc2+1)*500,ballc*3))
pred5003=model.predict(a3c.reshape((ballc-ballc2+1)*500,ballc*3))
pred5004=model.predict(a4c.reshape((ballc-ballc2+1)*500,ballc*3))
pred5005=model.predict(a5c.reshape((ballc-ballc2+1)*500,ballc*3))
pred5006=model.predict(a6c.reshape((ballc-ballc2+1)*500,ballc*3))            
pred5007=model.predict(a7c.reshape((ballc-ballc2+1)*500,ballc*3))
pred5008=model.predict(a8c.reshape((ballc-ballc2+1)*500,ballc*3))
bs1=slash(b1c,100,bangle1)
bs2=slash(b2c,100,bangle2)
bs3=slash(b3c,100,bangle3)
bs4=slash(b4c,100,bangle4)
bs5=slash(b5c,100,bangle5)
bs6=slash(b6c,100,bangle6)
bs7=slash(b7c,100,bangle7)
bs8=slash(b8c,100,bangle8)
print(pred5001.shape)'''

errangle1=pred1-bangle1#計算預測資料與實際資料的誤差
errangle2=pred2-bangle2
errangle3=pred3-bangle3
errangle4=pred4-bangle4
errangle5=pred5-bangle5
errangle6=pred6-bangle6
errangle7=pred7-bangle7
errangle8=pred8-bangle8
errangle=np.vstack([errangle1,errangle2,errangle3,errangle4,errangle6,errangle7,errangle8])

print(np.mean(np.abs(errangle[:,0])), np.std(errangle[:,0]), np.std(np.abs(errangle[:,0])))#計算平均誤差與標準差
print(np.mean(np.abs(errangle[:,1])), np.std(errangle[:,1]), np.std(np.abs(errangle[:,1])))
plt.figure(2)
plt.figure(figsize=(8,6))
plt.plot(errangle[:,0],errangle[:,1],'o')
plt.show()

def pole3d(errangle):
  row,col=errangle.shape
  count=np.zeros([9,7])
  for i in range(row):
    if round(errangle[i,0])<-20:
      count_x=0
    elif round(errangle[i,0])<-10 and round(errangle[i,0])>=-20:
      count_x=1
    elif round(errangle[i,0])<0 and round(errangle[i,0])>=-10:
      count_x=2
    elif round(errangle[i,0])<10 and round(errangle[i,0])>=0:
      count_x=3
    elif round(errangle[i,0])<20 and round(errangle[i,0])>=10:
      count_x=4
    elif round(errangle[i,0])<30 and round(errangle[i,0])>=20:
      count_x=5
    elif round(errangle[i,0])<40 and round(errangle[i,0])>=30:
      count_x=6
    if round(errangle[i,1])<-30:
      count_y=0
    elif round(errangle[i,1])<-20 and round(errangle[i,1])>=-30:
      count_y=1
    elif round(errangle[i,1])<-10 and round(errangle[i,1])>=-20:
      count_y=2
    elif round(errangle[i,1])<0 and round(errangle[i,1])>=-10:
      count_y=3
    elif round(errangle[i,1])<10 and round(errangle[i,1])>=0:
      count_y=4
    elif round(errangle[i,1])<20 and round(errangle[i,1])>=10:
      count_y=5
    elif round(errangle[i,1])<30 and round(errangle[i,1])>=20:
      count_y=6
    elif round(errangle[i,1])<40 and round(errangle[i,1])>=30:
      count_y=7
    elif round(errangle[i,1])<50 and round(errangle[i,1])>=40:
      count_y=8
    count[count_y,count_x]=count[count_y,count_x]+1
  return count

# 绘图设置
fig = plt.figure(figsize=(8,10))

ax = fig.gca(projection='3d')  # 三维坐标轴
X = [30,20,10,0,-10,-20,-30]
Y = [-40,-30,-20,-10,-0,10,20,30,40]
Z = pole3d(errangle) # 生成16个随机整数
Z = Z.ravel()

# meshgrid把X和Y变成平方长度，比如原来都是4，经过meshgrid和ravel之后，长度都变成了16，因为网格点是16个
xx, yy = np.meshgrid(X, Y)  # 网格化坐标
X, Y = xx.ravel(), yy.ravel()  # 矩阵扁平化
# 设置柱子属性
height = np.zeros_like(Z) # 新建全0数组，shape和Z相同
width = depth = 10 # 柱子的长和宽
# 颜色数组，长度和Z一致
ax.invert_yaxis() 
# 开始画图，注意本来的顺序是X, Y, Z, width, depth, height，但是那样会导致不能形成柱子，只有柱子顶端薄片，所以Z和height要互换
ax.bar3d(X, Y, height, width, depth, Z,  color='lightgreen', shade=True,edgecolor='black')  # width, depth, height
ax.set_xlabel('Δθv')
ax.set_ylabel('Δθh')
plt.show()

def mixa(tra1,tra2,w1,w2,sample,count2):#將兩個軌跡結合
            t1=np.zeros([sample,w1])
            x1=np.zeros([sample,w1])
            y1=np.zeros([sample,w1])
            z1=np.zeros([sample,w1])
            t2=np.zeros([sample,w2])
            x2=np.zeros([sample,w2])
            y2=np.zeros([sample,w2])
            z2=np.zeros([sample,w2])
            count=np.zeros(sample)
            count2=np.zeros([sample,count2])
            for a in range(sample):#t1為第2組軌跡
                for i in range(w1):
                    t1[a,i]=tra1[a,4*i]
                    x1[a,i]=tra1[a,4*i+1]
                    y1[a,i]=tra1[a,4*i+2]
                    z1[a,i]=tra1[a,4*i+3]
                for i in range(w2):#t2為第1組軌跡
                    t2[a,i]=tra2[a,4*i]
                    x2[a,i]=tra2[a,4*i+1]
                    y2[a,i]=tra2[a,4*i+2]
                    z2[a,i]=tra2[a,4*i+3]
                for i in range(w2):#i為第二組雙眼視覺擷取到軌跡前的桌球軌跡數
                    if t2[a,i]==t1[a,0]:
                        #print(i)
                        break
                for j in range(w2-i):#j為兩組雙眼視覺都有捉到球的軌跡數
                    if t2[a,i+j]!=t1[a,j]:
                        #print(i+j-1)
                        break
                #print(w1+i)
                t3=np.zeros([w1+i])
                x3=np.zeros([w1+i])
                y3=np.zeros([w1+i])
                z3=np.zeros([w1+i])
                t4=np.zeros([w1+i])
                x4=np.zeros([w1+i])
                y4=np.zeros([w1+i])
                z4=np.zeros([w1+i])
                t3[:i+j]=t2[a,:i+j]
                x3[:i+j]=x2[a,:i+j]
                y3[:i+j]=y2[a,:i+j]
                z3[:i+j]=z2[a,:i+j]
                t3[i+j:]=t1[a,j:]
                x3[i+j:]=x1[a,j:]
                y3[i+j:]=y1[a,j:]
                z3[i+j:]=z1[a,j:]#t3的重疊部分是以第1組雙眼視覺為主
                t4[:i]=t2[a,:i]
                x4[:i]=x2[a,:i]
                y4[:i]=y2[a,:i]
                z4[:i]=z2[a,:i]
                t4[i:]=t1[a]
                x4[i:]=x1[a]
                y4[i:]=y1[a]
                z4[i:]=z1[a]#t4的重疊部分是以第2組雙眼視覺為主
                x5=x3
                y5=y3
                z5=z3
                x5[i+j:]=x3[i+j:]+np.average((x3-x4)[i+j-5:i+j])
                y5[i+j:]=y3[i+j:]+np.average((y3-y4)[i+j-5:i+j])
                z5[i+j:]=z3[i+j:]+np.average((z3-z4)[i+j-5:i+j])#t5是把1和2組平均後把1加上平均誤差(以第1組為主，每5顆為一平均誤差)
                #print(np.average((y3-y4)[i+j-5:i+j]))
                
                for i in range(len(y5)):#找反彈點
                    if y5[i]<y5[i+1]:
                        break
                #print(y5)
                #print(y5[:i+1])
                tra=np.zeros([len(y5[:i+1])*3])
                for j in range(len(y5[:i+1])):
                    tra[3*j]=x5[j]
                    tra[3*j+1]=y5[j]
                    tra[3*j+2]=z5[j]
                #print(tra.reshape(len(y5[:i+1]),3))
                count[a]=len(tra)
                count2[a,:len(tra)]=tra
                #print(count)
            return count2
tra1=mixa(rback5_500,back5_500,rw1,w1,500,141)
tra2=mixa(rleft5_500,left5_500,rw2,w2,500,156)
tra3=mixa(rright5_500,right5_500,rw3,w3,500,138)
tra4=mixa(rtop5_500,top5_500,rw4,w4,500,120)
tra5=mixa(rleftback5_500,leftback5_500,rw5,w5,500,141)
tra6=mixa(rlefttop5_500,lefttop5_500,rw6,w6,500,126)
tra7=mixa(rrightback5_500,rightback5_500,rw7,w7,500,123)
tra8=mixa(rrighttop5_500,righttop5_500,rw8,w8,500,123)
for i in range(500):
   if tra3[i,137] != 0:
     break
print(tra3[i])
'''
def mixnew(normal,rebound,w1,rw1):
  combine = np.zeros([500,(w1+rw1)*4])
  combine2 = np.zeros([500,(w1+rw1)*3])
  x = np.zeros([w1+rw1])
  y = np.zeros([w1+rw1])
  z = np.zeros([w1+rw1])
  for i2 in range(500): 
    for i in range(int(w1/4)):
      if normal[i2,i*4] == 0:
        break
    b = normal[i2,(i-1)*4]
    for j in range(int(rw1/4)):
      if  rebound[i2,j*4] == b:
        break   
    combine[i2,:(i-1)*4] = normal[i2,:(i-1)*4]
    combine[i2,(i-1)*4:(i-1)*4+(len(rebound[i2])-j*4)] = rebound[i2,j*4:len(rebound[i2])]
    for k in range(w1+rw1):
      x[k]=combine[i2,k*4+1]
      y[k]=combine[i2,k*4+2]
      z[k]=combine[i2,k*4+3]
    for k in range(w1+rw1):
      if y[k]<y[k+1]:
        break      
    for m in range(k+1):
      combine2[i2,m*3]=x[m]
      combine2[i2,m*3+1]=y[m]
      combine2[i2,m*3+2]=z[m]
  return combine2,combine
totalback,total= mixnew(back5_500,rback5_500,w1,rw1)
print(totalback)'''

tes1=mixa(rback5_100,back5_100,rw1,w1,100,141)
tes2=mixa(rleft5_100,left5_100,rw2,w2,100,156)
tes3=mixa(rright5_100,right5_100,rw3,w3,100,138)
tes4=mixa(rtop5_100,top5_100,rw4,w4,100,120)
tes5=mixa(rleftback5_100,leftback5_100,rw5,w5,100,141)
tes6=mixa(rlefttop5_100,lefttop5_100,rw6,w6,100,126)
tes7=mixa(rrightback5_100,rightback5_100,rw7,w7,100,123)
tes8=mixa(rrighttop5_100,righttop5_100,rw8,w8,100,123)
print(tes1)

ball=9#47=141/3
nw1=47-ball*2+1
nw2=49-ball*2+1
nw3=46-ball*2+1
nw4=40-ball*2+1
nw5=47-ball*2+1
nw6=42-ball*2+1
nw7=41-ball*2+1
nw8=41-ball*2+1
print(s1.shape)
'''
def trainnew(tra,n,w,s,k):
            train=np.zeros([500,n,3])
            for i in range(500):#整理軌跡
                for j in range(n):
                    x=[]
                    y=[]
                    z=[]
                    x.append(tra[i,0+3*j])
                    y.append(tra[i,1+3*j])
                    z.append(tra[i,2+3*j])
                    x=np.array(x)    
                    y=np.array(y)
                    z=np.array(z)
                    train[i,j,0]=x
                    train[i,j,1]=y
                    train[i,j,2]=z
            before=np.zeros([w*500,ball,3])
            after=np.zeros([w*500,ball,3])
            sla=np.zeros([w*500,3])
            a=0
            c=0
            flag = 0
            #print(train[0,29])
            for i1 in range(500):
              flag = 0
              for j1 in range(w):
                if train[i1,j1+ball:j1+ball*2][k,0] == 0:
                  flag = 1
                  break  
                before[j1+a] = train[i1,j1:j1+ball]
                after[j1+a] = train[i1,j1+ball:j1+ball*2]
                sla[j1+a] = s[i1]
              if flag == 0:
                a+=j1+1
              else:
                a+=j1       
            for i2 in range(w*500):
              if after[i2,0,0] == 0:
                before2 = before[:i2]
                after2 = after[:i2]
                sla2 = sla[:i2]
                break    
            return before2,after2,sla2,a
newa1,newb1,sla1,count1=trainnew(tra1,47,nw1,s1,2)
newa2,newb2,sla2,count2=trainnew(tra2,49,nw2,s2,2)
newa3,newb3,sla3,count3=trainnew(tra3,46,nw3,s3,2)
newa4,newb4,sla4,count4=trainnew(tra4,40,nw4,s4,2)
newa5,newb5,sla5,count5=trainnew(tra5,47,nw5,s5,2)
newa6,newb6,sla6,count6=trainnew(tra6,42,nw6,s6,2)
newa7,newb7,sla7,count7=trainnew(tra7,41,nw7,s7,3)
newa8,newb8,sla8,count8=trainnew(tra8,41,nw8,s8,3)
print(sla8.shape)'''

def train(tra500,n,w,point_500,k):#in:總軌跡 out:從第一個軌跡漸進1個軌跡的桌球軌跡  從第9個軌跡漸進1個軌跡的桌球軌跡 1-10刻的斜率與水平和垂直角度差組合
            train=np.zeros([500,n,3])
            c=0
            for i in range(500):#整理軌跡
                for j in range(n):
                    x=[]
                    y=[]
                    z=[]
                    x.append(tra500[i,0+3*j])
                    y.append(tra500[i,1+3*j])
                    z.append(tra500[i,2+3*j])
                    x=np.array(x)    
                    y=np.array(y)
                    z=np.array(z)
                    train[i,j,0]=x
                    train[i,j,1]=y
                    train[i,j,2]=z
            a1=np.zeros([w*500,ball,3])
            b1=np.zeros([w*500,ball,3])
            a=0
            for j in range(500):
                for i in range(w):
                    if train[j,i+ball:i+ball*2][k,0] == 0:
                        a+=(j+1)*w-(i+w*j)
                        #print(a)
                        break
                    a1[i+w*j-a]=train[j,i:i+ball]
                    b1[i+w*j-a]=train[j,i+ball:i+ball*2]
            for i2 in range(10):
                if b1[i+w*j-a+i2,0,0] == 0:
                    aa1=a1[:i+w*j-a+i2]
                    bb1=b1[:i+w*j-a+i2]
                    #print(i+w*j-a+i2)#一種球有多少漸近式軌跡
                    break
             
            #a11=np.zeros([i+w*j-a+i2,2])
            a11=np.zeros([i+w*j-a+i2,3])
            #a11=np.zeros([i+w*j-a+i2,4])
            a=0
            for j in range(500):
                for i in range(w):
                    if train[j,i+ball:i+ball*2][k,0] == 0:
                        a+=(j+1)*w-(i+w*j)
                        break
                    a11[i+w*j-a]=point_500[j]
              
            return aa1,bb1,a11
# 48 56 49 43 47 47 45 44
'''a1,b1,a11=train(tra1,47,nw1,angle1,2)
a2,b2,a22=train(tra2,49,nw2,angle2,2)
a3,b3,a33=train(tra3,46,nw3,angle3,2)
a4,b4,a44=train(tra4,40,nw4,angle4,2)
a5,b5,a55=train(tra5,47,nw5,angle5,2)
a6,b6,a66=train(tra6,42,nw6,angle6,2)
a7,b7,a77=train(tra7,41,nw7,angle7,3)
a8,b8,a88=train(tra8,41,nw8,angle8,3)'''
a1,b1,a11=train(tra1,47,nw1,s1,2)
a2,b2,a22=train(tra2,49,nw2,s2,2)
a3,b3,a33=train(tra3,46,nw3,s3,2)
a4,b4,a44=train(tra4,40,nw4,s4,2)
a5,b5,a55=train(tra5,47,nw5,s5,2)
a6,b6,a66=train(tra6,42,nw6,s6,2)
a7,b7,a77=train(tra7,41,nw7,s7,3)
a8,b8,a88=train(tra8,41,nw8,s8,3)
print(a88.shape)
'''a1,b1,a11=train(tra1,47,nw1,pred5001,2)
a2,b2,a22=train(tra2,49,nw2,pred5002,2)
a3,b3,a33=train(tra3,46,nw3,pred5003,2)
a4,b4,a44=train(tra4,40,nw4,pred5004,2)
a5,b5,a55=train(tra5,47,nw5,pred5005,2)
a6,b6,a66=train(tra6,42,nw6,pred5006,2)
a7,b7,a77=train(tra7,41,nw7,pred5007,3)
a8,b8,a88=train(tra8,41,nw8,pred5008,3)'''

g1=np.zeros([107516,9,3])#將軌跡做成資料集
g1[:14829]=a1
g1[14829:30749]=a2
g1[30749:44829]=a3
g1[44829:56327]=a4
g1[56327:71184]=a5
g1[71184:83678]=a6
g1[83678:95519]=a7
g1[95519:]=a8
h1=np.zeros([107516,9,3])
h1[:14829]=b1
h1[14829:30749]=b2
h1[30749:44829]=b3
h1[44829:56327]=b4
h1[56327:71184]=b5
h1[71184:83678]=b6
h1[83678:95519]=b7
h1[95519:]=b8
p=np.vstack([a11,a22,a33,a44,a55,a66,a77,a88])
print(p.shape)

def test(tra250,n,w,point_100):
            test=np.zeros([100,n,3])
            for i in range(100):
                for j in range(n):
                    x=[]
                    y=[]
                    z=[]
                    x.append(tra250[i,0+3*j])
                    y.append(tra250[i,1+3*j])
                    z.append(tra250[i,2+3*j])
                    x=np.array(x)    
                    y=np.array(y)
                    z=np.array(z)
                    test[i,j,0]=x
                    test[i,j,1]=y
                    test[i,j,2]=z
            c1=np.zeros([w*100,ball,3])
            d1=np.zeros([w*100,ball,3])
            c11=np.zeros([w*100,3])#訓練集尺寸修改點
            for j in range(100):
                for i in range(w):
                    c1[i+w*j]=test[j,i:i+ball]
                    d1[i+w*j]=test[j,i+ball:i+ball*2]
                    c11[i+w*j]=point_100[j]
            return c1,d1,c11
'''c1,d1,c11=test(tes1,47,nw1,bangle1)
c2,d2,c22=test(tes2,49,nw2,bangle2)
c3,d3,c33=test(tes3,46,nw3,bangle3)
c4,d4,c44=test(tes4,40,nw4,bangle4)
c5,d5,c55=test(tes5,47,nw5,bangle5)
c6,d6,c66=test(tes6,42,nw6,bangle6)
c7,d7,c77=test(tes7,41,nw7,bangle7)
c8,d8,c88=test(tes8,41,nw8,bangle8)'''
'''c1,d1,c11=test(tes1,47,nw1,pred1)
c2,d2,c22=test(tes2,49,nw2,pred2)
c3,d3,c33=test(tes3,46,nw3,pred3)
c4,d4,c44=test(tes4,40,nw4,pred4)
c5,d5,c55=test(tes5,47,nw5,pred5)
c6,d6,c66=test(tes6,42,nw6,pred6)
c7,d7,c77=test(tes7,41,nw7,pred7)
c8,d8,c88=test(tes8,41,nw8,pred8)'''
c1,d1,c11=test(tes1,47,nw1,bs1)
c2,d2,c22=test(tes2,49,nw2,bs2)
c3,d3,c33=test(tes3,46,nw3,bs3)
c4,d4,c44=test(tes4,40,nw4,bs4)
c5,d5,c55=test(tes5,47,nw5,bs5)
c6,d6,c66=test(tes6,42,nw6,bs6)
c7,d7,c77=test(tes7,41,nw7,bs7)
c8,d8,c88=test(tes8,41,nw8,bs8)

input1= Input(shape=(3,9))
input2= Input(shape=(3,1))

t=LSTM(400)(input1)
t2=LSTM(400)(input2)
add=concatenate([t, t2])
#add=Lambda(lambda x: x[0]+x[1])([t, t2])
#add=Subtract()([t, t2])
t=RepeatVector(3)(add)
t=ELU(0.2)(t)
t=LSTM(400, return_sequences=True)(t)

output1=TimeDistributed(Dense(9))(t)
model2=Model(inputs=[input1,input2],outputs=output1)
model2.compile(optimizer='adam', loss='mse')
#from keras.utils.vis_utils import plot_model
#model2=load_model('/content/drive/MyDrive/pred/pred0119_3.h5')#12_4  24_5  24_6
#plot_model(model2, to_file='/content/drive/MyDrive/classified/model2.png',show_shapes = True)
#model2.fit([g1.reshape(107516,3,ball),p.reshape(107516,3,1)],h1.reshape(107516,3,ball),batch_size=500,epochs=500)
history = model2.fit([g1.reshape(107516,3,ball),p.reshape(107516,3,1)],h1.reshape(107516,3,ball),batch_size=1000,epochs=400)
model2.save('C:/Users/liyuchen/Desktop/504/tabletennies_model/pred0614_3.h5')
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

dd1=np.zeros([100,nw1+ball-1,3])#
for n in range(0,nw1*100,nw1):
  dd1[int(n/nw1)]=np.vstack((d1[n],d1[n+ball],d1[n+ball*2],d1[n+ball*3],d1[n+nw1-1,ball-(nw1-1-3*ball):ball]))
dd2=np.zeros([100,nw2+ball-1,3])
for n in range(0,nw2*100,nw2):
  dd2[int(n/nw2)]=np.vstack((d2[n],d2[n+ball],d2[n+ball*2],d2[n+ball*3],d2[n+nw2-1,ball-(nw2-1-3*ball):ball]))
dd3=np.zeros([100,nw3+ball-1,3])
for n in range(0,nw3*100,nw3):
  dd3[int(n/nw3)]=np.vstack((d3[n],d3[n+ball],d3[n+ball*2],d3[n+ball*3],d3[n+nw3-1,ball-(nw3-1-3*ball):ball]))
dd4=np.zeros([100,nw4+ball-1,3])
for n in range(0,nw4*100,nw4):
  dd4[int(n/nw4)]=np.vstack((d4[n],d4[n+ball],d4[n+ball*2],d4[n+nw4-1,ball-(nw4-1-2*ball):ball]))
dd5=np.zeros([100,nw5+ball-1,3])
for n in range(0,nw5*100,nw5):
  dd5[int(n/nw5)]=np.vstack((d5[n],d5[n+ball],d5[n+ball*2],d5[n+ball*3],d5[n+nw5-1,ball-(nw5-1-3*ball):ball]))
dd6=np.zeros([100,nw6+ball-1,3])
for n in range(0,nw6*100,nw6):
  dd6[int(n/nw6)]=np.vstack((d6[n],d6[n+ball],d6[n+ball*2],d6[n+nw6-1,ball-(nw6-1-2*ball):ball]))
dd7=np.zeros([100,nw7+ball-1,3])
for n in range(0,nw7*100,nw7):
  dd7[int(n/nw7)]=np.vstack((d7[n],d7[n+ball],d7[n+ball*2],d7[n+nw7-1,ball-(nw7-1-2*ball):ball]))
dd8=np.zeros([100,nw8+ball-1,3])
for n in range(0,nw8*100,nw8):
  dd8[int(n/nw8)]=np.vstack((d8[n],d8[n+ball],d8[n+ball*2],d8[n+nw8-1,ball-(nw8-1-2*ball):ball]))
print(dd2[3].shape)

def pre1(c1,c11,dd1,k,k2,w1):
            y_last=-35
            print(c1[w1*k+k2])
            #o=model2.predict([c1[w1*k+k2].reshape(1,ball,3),c11[w1*k+k2].reshape(1,2,1)],verbose=1).reshape(1,ball,3)
            o=model2.predict([c1[w1*k+k2].reshape(1,3,ball),c11[w1*k+k2].reshape(1,3,1)],verbose=1).reshape(1,ball,3)
            i=1
            q1=0
            while i < ball+1 :
                if o[:,i-1,1] < y_last:
                    break
                elif i == ball:
                    o2=o
                    print(o)
                    #o=model2.predict([o.reshape(1,ball,3),c11[w1*k+k2].reshape(1,2,1)],verbose=1).reshape(1,ball,3)
                    o=model2.predict([o.reshape(1,3,ball),c11[w1*k+k2].reshape(1,3,1)],verbose=1).reshape(1,ball,3)
                    i=0
                    q1+=1
                i+=1
            print(o)
            i=0
            while i < ball :
                if o[:,i,1] < y_last:
                    break
                i+=1
            q2=i
            print(i)
            if i != 0:
                ox45=(y_last-o[:,i-1,1])/(o[:,i,1]-o[:,i-1,1])*(o[:,i,0]-o[:,i-1,0])+o[:,i-1,0]
                oz45=(y_last-o[:,i-1,1])/(o[:,i,1]-o[:,i-1,1])*(o[:,i,2]-o[:,i-1,2])+o[:,i-1,2]
                q3=(y_last-o[:,i-1,1])/(o[:,i,1]-o[:,i-1,1])*(1/60)
            else:
                ox45=(y_last-o2[:,ball-1,1])/(o[:,i,1]-o2[:,ball-1,1])*(o[:,i,0]-o2[:,ball-1,0])+o2[:,ball-1,0]
                oz45=(y_last-o2[:,ball-1,1])/(o[:,i,1]-o2[:,ball-1,1])*(o[:,i,2]-o2[:,ball-1,2])+o2[:,ball-1,2]
                q3=(y_last-o2[:,ball-1,1])/(o[:,i,1]-o2[:,ball-1,1])*(1/60)
            i=0
            while i < len(dd1[0]) :
                if dd1[int(k),i,1]<y_last:
                    break
                else:
                    i+=1
            q4=i
            dx45=(y_last-dd1[int(k),i-1,1])/(dd1[int(k),i,1]-dd1[int(k),i-1,1])*(dd1[int(k),i,0]-dd1[int(k),i-1,0])+dd1[int(k),i-1,0]
            dz45=(y_last-dd1[int(k),i-1,1])/(dd1[int(k),i,1]-dd1[int(k),i-1,1])*(dd1[int(k),i,2]-dd1[int(k),i-1,2])+dd1[int(k),i-1,2]
            q6=(y_last-dd1[int(k),i-1,1])/(dd1[int(k),i,1]-dd1[int(k),i-1,1])*(1/60)
            q5=-1
            for i in range(q4):
                if dd1[int(k),i,1] == c1[w1*k+k2,8,1] :
                    q5=i
                    break

            print(dx45,dz45)
            print(ox45,oz45)
            print(ox45-dx45,oz45-dz45)
            print(dd1[k])
            print(c11[w1*k+k2])
            print((q1*ball+q2)/60+q3-((q4-q5-1)/60+q6))
#pre1(c1,c11,dd1,10,8,nw1) #第幾筆, 第n~n+9顆
#pre1(c2,c22,dd2,80,2,nw2)
#pre1(c8,c88,dd8,20,8,nw8)

from tqdm import tqdm, trange

def predict(c1,c11,dd1,w):#將測試集導入模型進行測試
            ball30=20
            xerror_back=np.zeros([ball30*100])
            zerror_back=np.zeros([ball30*100])
            time=np.zeros([ball30*100])
            mean_xerror_back=np.zeros([ball30])
            mean_zerror_back=np.zeros([ball30])
            mean_time=np.zeros([ball30])
            p=w
            k=0
            k1=0
            y_last=-35
            progress = tqdm(total=20)
            for j2 in range(ball30):
                for j in range(0,p*100,p):
                    #print(j)
                    q1=0
                    #o=model2.predict([c1[j+j2].reshape(1,3,ball),c11[j+j2].reshape(1,2,1)],verbose=0).reshape(1,ball,3)
                    o=model2.predict([c1[j+j2].reshape(1,3,ball),c11[j+j2].reshape(1,3,1)],verbose=0).reshape(1,ball,3)
                    #o=model2.predict([c1[j+j2].reshape(1,3,ball),c11[j+j2].reshape(1,4,1)],verbose=0).reshape(1,ball,3)
                    #o=model.predict(c1[j+j2].reshape(1,ball,3),verbose=0).reshape(1,ball,3)
                    #print(o)
                    i=1
                    while i < ball+1 :
                        if o[:,i-1,1] < y_last:
                            #print(o[:,i,1])
                            break
                        elif i == ball:
                            o2=o
                            #o=model2.predict([o.reshape(1,3,ball),c11[j+j2].reshape(1,2,1)],verbose=0).reshape(1,ball,3)
                            o=model2.predict([o.reshape(1,3,ball),c11[j+j2].reshape(1,3,1)],verbose=0).reshape(1,ball,3)
                            #o=model2.predict([o.reshape(1,3,ball),c11[j+j2].reshape(1,4,1)],verbose=0).reshape(1,ball,3)
                            #o=model.predict(o.reshape(1,ball,3),verbose=0).reshape(1,ball,3)
                            #print(o)
                            i=0
                            q1+=1
                        i+=1
                    #print(d5[j+27])
                    #print(o)
                    i=0
                    while i < ball :
                        if o[:,i,1] < y_last:
                            break
                        i+=1
                    q2=i
                    if i != 0:
                        ox45=(y_last-o[:,i-1,1])/(o[:,i,1]-o[:,i-1,1])*(o[:,i,0]-o[:,i-1,0])+o[:,i-1,0]
                        oz45=(y_last-o[:,i-1,1])/(o[:,i,1]-o[:,i-1,1])*(o[:,i,2]-o[:,i-1,2])+o[:,i-1,2]
                        q3=(y_last-o[:,i-1,1])/(o[:,i,1]-o[:,i-1,1])*(1/60)
                    else:
                        ox45=(y_last-o2[:,ball-1,1])/(o[:,i,1]-o2[:,ball-1,1])*(o[:,i,0]-o2[:,ball-1,0])+o2[:,ball-1,0]
                        oz45=(y_last-o2[:,ball-1,1])/(o[:,i,1]-o2[:,ball-1,1])*(o[:,i,2]-o2[:,ball-1,2])+o2[:,ball-1,2]
                        q3=(y_last-o2[:,ball-1,1])/(o[:,i,1]-o2[:,ball-1,1])*(1/60)
                    i=0
                    while i < w :
                        if dd1[int(j/p),i,1] < y_last:
                            break
                        else:
                            i+=1
                    #print(i)
                    #print(j/p)
                    q4=i
                    dx45=(y_last-dd1[int(j/p),i-1,1])/(dd1[int(j/p),i,1]-dd1[int(j/p),i-1,1])*(dd1[int(j/p),i,0]-dd1[int(j/p),i-1,0])+dd1[int(j/p),i-1,0]
                    dz45=(y_last-dd1[int(j/p),i-1,1])/(dd1[int(j/p),i,1]-dd1[int(j/p),i-1,1])*(dd1[int(j/p),i,2]-dd1[int(j/p),i-1,2])+dd1[int(j/p),i-1,2]
                    q6=(y_last-dd1[int(j/p),i-1,1])/(dd1[int(j/p),i,1]-dd1[int(j/p),i-1,1])*(1/60)
                    q5=-1
                    for i in range(q4):
                        if dd1[int(j/p),i,1] == c1[j+j2,8,1] :
                            q5=i
                            break
                    #print(ox45,oz45)
                    #print(dx45,dz45)
                    #print(ox45-dx45,oz45-dz45)
                    xerror_back[k]=ox45-dx45
                    zerror_back[k]=oz45-dz45
                    time[k]=(q1*ball+q2)/60+q3-((q4-q5-1)/60+q6)
                    k+=1
                mean_xerror_back[k1]=np.mean(np.abs(xerror_back[k1*100:100+k1*100]))
                mean_zerror_back[k1]=np.mean(np.abs(zerror_back[k1*100:100+k1*100]))
                mean_time[k1]=np.mean(np.abs(time[k1*100:100+k1*100]))
                progress.update(1)
                k1+=1
            return mean_xerror_back,mean_zerror_back,xerror_back,zerror_back,time,mean_time
mean_xerror_back,mean_zerror_back,xerror_back,zerror_back,time_back,mean_time_back=predict(c1,c11,dd1,nw1)
print('======back predict done!!======')
mean_xerror_left,mean_zerror_left,xerror_left,zerror_left,time_left,mean_time_left=predict(c2,c22,dd2,nw2)
print('======left predict done!!======')
mean_xerror_right,mean_zerror_right,xerror_right,zerror_right,time_right,mean_time_right=predict(c3,c33,dd3,nw3)
print('======right predict done!!======')
mean_xerror_top,mean_zerror_top,xerror_top,zerror_top,time_top,mean_time_top=predict(c4,c44,dd4,nw4)
print('======top predict done!!======')
mean_xerror_leftback,mean_zerror_leftback,xerror_leftback,zerror_leftback,time_leftback,mean_time_leftback=predict(c5,c55,dd5,nw5)
print('======leftback predict done!!======')
mean_xerror_lefttop,mean_zerror_lefttop,xerror_lefttop,zerror_lefttop,time_lefttop,mean_time_lefttop=predict(c6,c66,dd6,nw6)
print('======lefttop predict done!!======')
mean_xerror_rightback,mean_zerror_rightback,xerror_rightback,zerror_rightback,time_rightback,mean_time_rightback=predict(c7,c77,dd7,nw7)
print('======rightback predict done!!======')
mean_xerror_righttop,mean_zerror_righttop,xerror_righttop,zerror_righttop,time_righttop,mean_time_righttop=predict(c8,c88,dd8,nw8)
print('======righttop predict done!!======')
print('======predict complete!!!======')

total_mean_xerror=(abs(mean_xerror_back)+abs(mean_xerror_left)+abs(mean_xerror_right)+abs(mean_xerror_top)+abs(mean_xerror_leftback)+abs(mean_xerror_lefttop)+abs(mean_xerror_rightback)+abs(mean_xerror_righttop))/8
print(total_mean_xerror)
total_mean_yerror=(abs(mean_zerror_back)+abs(mean_zerror_left)+abs(mean_zerror_right)+abs(mean_zerror_top)+abs(mean_zerror_leftback)+abs(mean_zerror_lefttop)+abs(mean_zerror_rightback)+abs(mean_zerror_righttop))/8
print(total_mean_yerror)
total_mean_terror=(abs(mean_time_back)+abs(mean_time_left)+abs(mean_time_right)+abs(mean_time_top)+abs(mean_time_leftback)+abs(mean_time_lefttop)+abs(mean_time_rightback)+abs(mean_time_righttop))/8
print(total_mean_terror)

merrx=np.zeros([20,100])#畫出x,zm與時間預測誤差圖
merrz=np.zeros([20,100])
merrt=np.zeros([20,100])
serrx=np.zeros([20,2])
serrz=np.zeros([20,2])
serrt=np.zeros([20,2])
def error_bar(k1):
    error_allx=(xerror_back[k1*100:100+k1*100]+xerror_left[k1*100:100+k1*100]+xerror_right[k1*100:100+k1*100]+xerror_top[k1*100:100+k1*100]+xerror_leftback[k1*100:100+k1*100]+xerror_lefttop[k1*100:100+k1*100]+xerror_rightback[k1*100:100+k1*100]+xerror_righttop[k1*100:100+k1*100])/8
    error_allz=(zerror_back[k1*100:100+k1*100]+zerror_left[k1*100:100+k1*100]+zerror_right[k1*100:100+k1*100]+zerror_top[k1*100:100+k1*100]+zerror_leftback[k1*100:100+k1*100]+zerror_lefttop[k1*100:100+k1*100]+zerror_rightback[k1*100:100+k1*100]+zerror_righttop[k1*100:100+k1*100])/8
    error_allt=(time_back[k1*100:100+k1*100]+time_left[k1*100:100+k1*100]+time_right[k1*100:100+k1*100]+time_top[k1*100:100+k1*100]+time_leftback[k1*100:100+k1*100]+time_lefttop[k1*100:100+k1*100]+time_rightback[k1*100:100+k1*100]+time_righttop[k1*100:100+k1*100])/8        
    quart_x=np.zeros(2)
    quart_z=np.zeros(2)
    quart_t=np.zeros(2)
    '''
    quart_x[0]=np.quantile(error_allx,0.25,interpolation='lower')
    quart_x[1]=np.quantile(error_allx,0.75,interpolation='higher')
    print(quart_x)
    '''
    #'''
    quart_z[0]=np.quantile(error_allz,0.25,interpolation='lower')
    quart_z[1]=np.quantile(error_allz,0.75,interpolation='higher')
    print(quart_z)
    #'''
    '''
    quart_t[0]=np.quantile(error_allt,0.25,interpolation='lower')
    quart_t[1]=np.quantile(error_allt,0.75,interpolation='higher')
    print(quart_t)
    '''
    return error_allx,error_allz,error_allt*1000

for i in range(20):
  merrx[i],merrz[i],merrt[i] = error_bar(i)
x=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
plt.figure(1,figsize=(9,4))
plt.title('X box plot', fontsize=14, fontweight='bold', fontname='FreeSerif')
plt.xlabel('update times', fontsize=14, fontname='FreeSerif')
plt.ylabel('error (cm)', fontsize=14, fontname='FreeSerif')
plt.xticks(range(1,21,1))
plt.grid(True)
plt.boxplot(merrx.transpose(),showfliers=False)
plt.figure(2,figsize=(9,4))
plt.title('Z box plot', fontsize=14, fontweight='bold', fontname='FreeSerif')
plt.xlabel('update times', fontsize=14, fontname='FreeSerif')
plt.ylabel('error (cm)', fontsize=14, fontname='FreeSerif')
plt.xticks(range(1,21,1))
plt.grid(True)
plt.boxplot(merrz.transpose(),showfliers=False)
plt.figure(3,figsize=(9,4))
plt.title('T box plot', fontsize=14, fontweight='bold', fontname='FreeSerif')
plt.xlabel('update times', fontsize=14, fontname='FreeSerif')
plt.ylabel('error (ms)', fontsize=14, fontname='FreeSerif')
plt.xticks(range(1,21,1))
plt.grid(True)
plt.boxplot(merrt.transpose(),showfliers=False)
plt.show()
