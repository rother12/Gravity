#                        ---------Module import,Species--------
import torch
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import visdom
import torchvision
import torchvision.transforms as tr
from sklearn.model_selection import train_test_split
import torch.nn.init
#                       ----------(1)Visdom 사용을 위한 loss_plt 정의------------
#                       ----------(1)define loss_plt to use Visdom------------
#                       (you must turn on visdom server[Typing in Anaconda [python -m visdom.server])
vis=visdom.Visdom()
def loss_tracker(loss_plot,loss_value,num):
    vis.line(X=num,Y=loss_value,win=loss_plot,update='append')
train_loss_plt=vis.line(Y=torch.Tensor(1).zero_(),opts=dict(xlabel='epoch',ylabel='Train_Loss',title='Train_loss',legend=['GP_CNN train_loss'],showlegend=True))
test_loss_plt=vis.line(Y=torch.Tensor(1).zero_(),opts=dict(xlabel='epoch',ylabel='Test_Loss',title='Validation_loss',legend=['GP_CNN test_loss'],showlegend=True))

#                      -----------(1-1)하이퍼 파러미터과 GPU 사용 세팅---------------
#                      -----------(1-1)Hyperparameter(save path=path) and Gpu setting---------------
#                      (At first, You Must Gpu setting in Your Computer, Search "Gpu setting in Pytorch")

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1004)

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
dataset=torchvision.datasets.ImageFolder(root=root,transform=tf1)

        #               -data split(train:0.7, test:0.3*0.5, validation:0.3*0.5)-
train_set,testdata=train_test_split(dataset,train_size=0.7,test_size=0.3,random_state=1004)
test_set,validation_set=train_test_split(testdata,train_size=0.5,test_size=0.5, random_state=1004)

        #               -splited data를 Model에 사용하기 편하게 조절(splited데이터를 Dataloader로)-
        #               -From splited data To Dataloader-
trainloader=torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True)
testloader=torch.utils.data.DataLoader(test_set,batch_size=batch_size,shuffle=False)
validationloader=torch.utils.data.DataLoader(validation_set,batch_size=batch_size,shuffle=False)

    #          ----(2-2)Model 정의(CNN2층,FC2층, Optimizer,criterion)----
    #          ----(2-2)Model define(CNN:2L,FC:2L, Optimizer,criterion)----

#              CNN:Convolution Neural Net, FC:Fully Connected(Linear Layer)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #                 -CNN 층 정의-
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
# (When you Change your Data Size you also change Libear node number)
    #        self.fc1 = nn.Linear(128 * 11 * 8, 128 * 2)        #  Single View size(47*57)
    #        self.fc1 = nn.Linear(128 * 20 * 25, 128 * 2)        # Merged Size (94*114)

    def forward(self, x):
        out = self.CNN1(x)
        out = self.CNN2(out)
        out=torch.flatten(out,1)
        out = self.FC1(out)
        out = self.FC2(out)
        return out

##        return F.softmax(out,dim=1)
##        return F.log_softmax(out, dim=1)

model= Net().to(device)
optimizer = optim.Adadelta(model.parameters(),lr=lr)
criterion=nn.CrossEntropyLoss().to(device)

#                            --------(3-1)Define Train 정의-------
def train(trainloader,model,optimizer,device,epoch,log_interval):
    model.train()
    for batch_idx,(data, label) in enumerate(trainloader):
        data, label = Variable(data).to(device), Variable(label).to(device)
        optimizer.zero_grad()
        output = model(data)
        loss=criterion(output,label)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                 epoch, batch_idx * len(data), len(trainloader.dataset),
                 100. * batch_idx / len(trainloader), loss.item()))

    loss_tracker(train_loss_plt, torch.Tensor([loss.item()]), torch.Tensor([epoch]))

#                            --------(3-2)Define Test 정의-------
def test(validationloader,model,device,epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, label in validationloader:
            data, label = Variable(data).to(device), Variable(label).to(device)
            output = model(data)
#                       ---------sum up batch loss 배치 loss를 더하는 과정-------
            test_loss += criterion(output, label).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(label.view_as(pred)).to(device).sum().item()
        test_loss /= len(validationloader.dataset)
        test_accuracy= 100.* correct/ len(validationloader.dataset)

        print('\nevaluation set: 평균 loss: {:.9f}, 정확도: {}/{} ({:.2f}%)\n'.format(test_loss,
            correct, len(validationloader.dataset),test_accuracy))

    loss_tracker(test_loss_plt,torch.Tensor([test_loss]),torch.Tensor([epoch]))

#                            --------(4-1) 실제 구동 및 모델 저장-------

for epoch in range(1,epoch+1):
    train(trainloader, model, optimizer, device, epoch,log_interval)
    test(validationloader, model, device, epoch)

    torch.save({
                  'epoch': epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  }, Path)

########################Comment for "G.S Train & Validation" code
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



    # 2.(G.S Train & Validation코드 참조)
#(1)Visdom 사용을 위한 loss_plt 정의
    #loss를 그리기 위해 Visdom을 사용하였습니다. 보통loss를 그릴때는 tensorboard를 이용합니다.
    #아나콘다에서 visdom서버를 열어야 합니다.
#(1-1)HyperParameter를 설정하고 Gpu사용을 위한 설정입니다.
    #Gpu사용이 가능하면 Gpu를 안되면 Cpu를 사용하게 코드 작성을 하였으나,속도를 위해 Torch의 Gpu사용 설정을 권합니다.
    #torch seed는 시행의 결과가  매번 달라짐을 막기 위해 설정하였습니다.
    # root=Data 로드 경로 이며 path는 save path로 지정하였음을 주의 합니다.
#(2-1)Data load와 Split
    #data split를 통해 train_set은 (0.7) testdata(0.3)으로 지정합니다.
    #testdata에 동일한 작업을 하여 validation_set(0.3*0.5)=(0.15) test_set(0.3*0.5)=(0.15)로 처리합니다.
    #여기서 test_Set은 G.S Test Z&Confusion Matrix에서 Test합니다.
#(2-2&3-1)
    #Net과 Train,Validation Model들을 정의합니다.