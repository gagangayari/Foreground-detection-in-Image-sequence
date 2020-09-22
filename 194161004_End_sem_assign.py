import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
#%%Read images

images=[cv2.imread(file) for file in glob.glob("F:/pdf e-reading/Mtech/2nd semester/Machine Learning/Lab/Final_lab_assignment/Images/*.bmp")]
backgroundtruth=[cv2.imread(gndtruth,0) for gndtruth in glob.glob("F:/pdf e-reading/Mtech/2nd semester/Machine Learning/Lab/Final_lab_assignment/BakSubGroundTruth-20200603T191438Z-001/BakSubGroundTruth/*.bmp")]

#%%Functions required
def kron_delta(x):
    if(x==0):
        return 1
    else:
        return 0

#alpha_t
def alpha_t(t):
    tc=100
    if t<=tc:
        return 1/t
    else:
        return 1/tc
def filter1(x):
    vC_temp=cv2.copyMakeBorder(x,2,2,2,2,borderType=cv2.BORDER_CONSTANT)
    #print(x.shape)
    #print(vC_temp.shape)

    for j in range(2,vC_temp.shape[0]-2):
        for k in range(2,vC_temp.shape[1]-2):
            temp1=vC_temp[j-2,k-2]+vC_temp[j-2,k-1]+vC_temp[j-2,k]+vC_temp[j-2,k+1]+vC_temp[j-2,k+2]
            temp2=vC_temp[j-1,k-2]+vC_temp[j-1,k-1]+vC_temp[j-1,k]+vC_temp[j-1,k+1]+vC_temp[j-1,k+2]
            temp3=vC_temp[j,k-2]+vC_temp[j,k-1]+vC_temp[j,k]+vC_temp[j,k+1]+vC_temp[j,k+2]
            temp4=vC_temp[j+1,k-2]+vC_temp[j+1,k-1]+vC_temp[j+1,k]+vC_temp[j+1,k+1]+vC_temp[j+1,k+2]
            temp5=vC_temp[j+2,k-2]+vC_temp[j+2,k-1]+vC_temp[j+2,k]+vC_temp[j+2,k+1]+vC_temp[j+2,k+2]
            temp=(temp1+temp2+temp3+temp4+temp5)/(255*25)
            if(temp>0.6): 
                x[j-2,k-2]=255             
            else:
               # z=x[j-2,k-2]
                x[j-2,k-2]=0
    return x


#%%
##Initializing 
lambda_=[4]


# Doing it for  first 50 images
no_of_imgs=22
from_slice=20
prev_image=images[from_slice-1]

true_label=backgroundtruth[from_slice:no_of_imgs]

TPR,FPR=[],[]
img_for_lmbda,imgs=[],[]

for lmbda in lambda_:
    TP,TN,FP,FN=0,0,0,0  
    predicted=[]
    V=np.zeros(images[0].shape)
    V[:,:,0]=9
    V[:,:,1]=9
    V[:,:,2]=9
    print('for lambda: ',lmbda)
    
    for im in range(from_slice,no_of_imgs):

        curr_image=images[im]  
        # cv2.imshow('',curr_image)
        # cv2.waitKey(3)
        D=curr_image-prev_image
        cv2.imshow('D',D)
    
        #####   Foreground Detection ###
        index1=(D**2)<=((lmbda**2)*V)  #indexes satisfying the chebysev condition
    
        C=np.ones((D.shape[0],D.shape[1]))
        
        for i in range(D.shape[0]):
            for j in range(D.shape[1]):
            
                if(index1[i][j][0]==index1[i][j][1]==index1[i][j][2]==True):
                    
                    C[i][j]=0
                else:
                    C[i][j]=255
        
        cv2.imshow('pred',C)
        cv2.imshow('tru',true_label[im-from_slice])
        # print('ok')~
        
        cv2.waitKey(0)
        ######   Noise removal ###
        
        #padding
        # vCt=cv2.copyMakeBorder(C,2,2,2,2,borderType=cv2.BORDER_REPLICATE)
        vCt=filter1(C)
        # m=5
        # kernel=np.ones((m,m))
        # for x in range(2,vCt.shape[0]-2):
        #     for y in range(2,vCt.shape[1]-2):
        #     # print(y)
        #         img_slice=vCt[x-2:x+3,y-2:y+3]
        #         conv=(kernel*img_slice).sum()
        #         no_of_pxels=conv/255
    
        #         if((no_of_pxels/25)>=0.6):
        #             vCt[x][y]=255
        #             C[x-2][y-2]=255
        #         else:
        #             vCt[x][y]=0
        #             C[x-2][y-2]=0
                    
        cv2.imshow('vCt',vCt)
        cv2.waitKey(0)

        
        
        #### Background Model Update ##
        for x_ in range(C.shape[0]):
            for y_ in range(C.shape[1]):
                for k in range(3):
                    # pass
                    
                    delta_kron1=kron_delta(C[x_][y_])
                    curr_image[x_][y_][k]=prev_image[x_][y_][k]+ (alpha_t(im-from_slice+1)*delta_kron1*D[x_][y_][k])
                    prev_image=curr_image
                    
                    # delta_kron2=kron_delta(C[x_][y_]-255)
                    # p=(1-alpha_t(im-from_slice+1))*((V[x_,y_,k])+(alpha_t(im+1-from_slice)*(D[x_,y_,k]**2)))
                    
                    # V[x_][y_][k]=(delta_kron2*V[x_][y_][k] ) + delta_kron1*p
            
        
        predicted.append(C)
        print('completed image :',im,'lambda :',lmbda)


    ####  Calculating Sensitivity and False alarms rate
    
    pred_img=np.asarray(predicted).ravel()
    true_img=np.asarray(true_label).ravel()
    
    TN,FP,FN,TP=confusion_matrix(true_img,pred_img).ravel()
    
        

    tpr = TP/(TP+FN)
    fpr = FP/(FP+TN)
    TPR.append(tpr)
    FPR.append(fpr)
    print('Done for lambda :',lmbda)
    
    
    
    

#%%plot ROC
plt.plot(FPR,TPR)
plt.title('ROC ')
plt.xlabel('False Positive Rate')
plt.ylabel('Sensitivity')
plt.show()

