import torch
from torch.utils.data import DataLoader,Dataset
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as tr
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import  sklearn.metrics as metrics

cmap0=plt.cm.Blues
species=('1080Lines','1400Ripples','Air_Compressor','Blip',
         'Chirp','Extremely_Loud','Helix','Koi_Fish','Light_Modulation',
         'Low_Frequency_Burst','Low_Frequency_Lines','No_Glitch',
         'None_of_the_Above','Paired_Doves','Power_Line','Repeating_Blips',
         'Scattered_Light','Scratchy','Tomte','Violin_Mode','Wandering_Line','Whistle')
classes=species

#                      -----------(1-1)하이퍼 파러미터과 GPU 사용 세팅---------------
#                      -----------(1-1)Hyperparameter(save path=path) and Gpu setting---------------
#                      (At first, You Must Gpu setting in Your Computer, Search "Gpu setting in Pytorch")

 #                     -Model Hyper parameter(Batch,lr,epoch,log_interval,Save path)-
batch_size=30
lr=0.2
epoch=50
log_interval=600

Path = "D://modelsGS//Saving origin_try004_5 soft.tar"
# Save path:path, Load path:root

#                     ----(2-1)Data load와 분할 작업 (root설정,split)-----
#                     ----(2-1)Data load define & split to train,test,validation set (root=Load Data path, split)-----

#root='D:\\GSorigin\\Just_resized1'
#root='D:\\GSorigin\\Just_resized2'
#root='D:\\GSorigin\\Just_resized3'
#root='D:\\GSorigin\\Just_resized4'
#root='D:\\GSorigin\\Just_resized_M'

#root='D:\\GSorigin\\Just_HSV1'
#root='D:\\GSorigin\\Just_HSV2'
#root='D:\\GSorigin\\Just_HSV3'
#root='D:\\GSorigin\\Just_HSV4'
root='D:\\GSorigin\\Just_HSV_M'

#               -data load(dataset을 Gray Scale로 ImageFolder를 이용하여 불러온다.)-
tf1=tr.Compose([tr.Grayscale(),tr.ToTensor()])


#                     ----(2-1)Data load와 분할 작업 (root설정,split)-----
#                     ----(2-1)Data load define & split to train,test,validation set (root=data path, split)-----

dataset=torchvision.datasets.ImageFolder(root=root,transform=tf1)

#-------------------------------------------------split 하는거
train_set,testdata=train_test_split(dataset,train_size=0.7,test_size=0.3)
test_set,validation_set=train_test_split(testdata,train_size=0.5,test_size=0.5)

#-----------------split 된 파일을 dataloader로 변환
trainloader=DataLoader(train_set, batch_size=batch_size ,shuffle=True)
testloader=DataLoader(test_set, batch_size=batch_size ,shuffle=False)
validationloader=DataLoader(validation_set, batch_size=batch_size ,shuffle=False)

    #          ----(2-2)Model 정의(CNN2층,FC2층, Optimizer,criterion)----
    #          ----(2-2)Model define(CNN:2L,FC:2L, Optimizer,criterion)----

#              CNN:Convolution Neural Net, FC:Fully Connected(Linear Layer)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.CNN1= torch.nn.Sequential(torch.nn.Conv2d(in_channels=1,out_channels=128, kernel_size=5, stride=1)
                                      ,torch.nn.MaxPool2d(kernel_size=2, stride=2)
                                      ,nn.ReLU())
        self.CNN2= torch.nn.Sequential(torch.nn.Conv2d(in_channels=128,out_channels=128, kernel_size=5, stride=1)
                                      ,torch.nn.MaxPool2d(kernel_size=2, stride=2)
                                      ,nn.ReLU())

        #                -FC 층 정의-
        self.FC1=torch.nn.Sequential(nn.Linear(128 * 20 * 25, 128 * 2)
                                      ,nn.ReLU())

        self.FC2=nn.Linear(128 *2 ,22)
#(When you Change your Data Size you also change Linear node number)

    #        self.fc1 = nn.Linear(128 * 11 * 8, 128 * 2)        # single view size(47*57)
    #        self.fc1 = nn.Linear(128 * 20 * 25, 128 * 2)        # merged view size(94*114)

    def forward(self, x):
#        in_size = x.size(0)
        out=self.CNN1(x)
        out=self.CNN2(out)
        out = torch.flatten(out, 1)
        out = self.FC1(out)
        out = self.FC2(out)
        return out

#        return F.softmax(x,dim=1)
#        return F.log_softmax(x,dim=1)

model= Net().to(device)
optimizer = optim.Adadelta(model.parameters(), lr=lr )#,momentum=momentum)#,lr=0.01

#-----load saved model
Path = "D://modelsGS//Saving origin_try003 soft.tar"

checkpoint = torch.load(Path)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
model.eval()


print("epoch:",epoch)

cmap=plt.cm.Blues
#----------------------------------- def Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=True,
                          title=None,
                          cmap=cmap):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
# Compute confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis])
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='true label',
           xlabel='predicted label')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
#             ,rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
#plot_confusion_matrix(true, pred, classes= classes,title='Confusion matrix, without normalization')

#----------define modeltest
def modeltest():
    y_t=[]
    y_p=[]
    test_loss = 0
    correct = 0
    for data, target in testloader:
        data, target = Variable(data).to(device), Variable(target).to(device)
        output = model(data).to(device)

        # sum up batch loss 배치 loss를 더하는 과정
        test_loss += F.cross_entropy(output, target, reduction='mean').data.to(device)
        pred = output.data.max(1, keepdim=True)[1]
        pred=pred.view_as(target)

        y_pred=pred.cpu()
        y_t.extend(target)
        y_p.extend(y_pred)


        correct += pred.eq(target.view_as(pred)).to(device).sum().item()
#        print(y_t.type,y_p.type)


        test_loss /= len(testloader.dataset)

    print('\ntest set: 평균 loss: {:.4f}, 정확도: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))

    correct += pred.eq(target.view_as(pred)).to(device).sum().item()


    y_t=torch.as_tensor(y_t)
    y_p=torch.as_tensor(y_p)

    plot_confusion_matrix(y_t, y_p, classes,
                      normalize=False,
                      title=None,
                      cmap=plt.cm.Blues)
    plt.show()

    plot_confusion_matrix(y_t, y_p, classes,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues)
    plt.show()

modeltest()

########################Comment for "G.s Test & confusion matrix" code
# 1.(Data preparation 코드 참조)
#   Data 전처리를 사용한다.
#   #(Module을 install해야 하며,Dataset을 다운받아야 합니다.)

# 2.(G.S Train & Validation코드 참조)
#    G.S(Zevin) & Deep Multi-View Models(Sara)모델을 만든후, 전처리 된 Data를 이용하여 Train과 Validataion을 check 합니다.
##   (코드를 사용하기 전에 파이토치에서 Pytorch에서 Gpu사용을 위한 설정과
##   전처리 된 Data의 path를 정확하게 설정하고 Single과 Merged에 따라 net의 구조가 달라짐을 유의합니다. )

# 3.(G.s Test & confusion matrix 코드 참조)
#   2에서 학습된 saved model을 가져와 test set에 대해 학습시키고, confusion matrix를 그립니다.
##   (이때,model의 변수가 동일한지와 save된 train data가 잘 맞는지 확인합니다.)
