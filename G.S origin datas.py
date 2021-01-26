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
#                       ----------Visdom을 사용을 위한 loss_plt 정의------------
vis=visdom.Visdom()
def loss_tracker(loss_plot,loss_value,num):
    vis.line(X=num,Y=loss_value,win=loss_plot,update='append')
train_loss_plt=vis.line(Y=torch.Tensor(1).zero_(),opts=dict(xlabel='epoch',ylabel='Train_Loss',title='Train_loss',legend=['GP_CNN train_loss'],showlegend=True))
test_loss_plt=vis.line(Y=torch.Tensor(1).zero_(),opts=dict(xlabel='epoch',ylabel='Test_Loss',title='Validation_loss',legend=['GP_CNN test_loss'],showlegend=True))

#                      -----------하이퍼 파러미터과 GPU 사용 세팅---------------
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#torch.manual_seed(1004)

 #                     -Model Hyper parameter(Batch,lr,epoch,log_interval)-
batch_size=30

lr=0.2
epoch=50

log_interval=600#150#600#200 #300 #400

##Path = "D://modelsGS//Saving origin_try004_1 soft.tar"  # 정확도 96.19%, root=justresizedM,LR=0.1batch_size=30,epoch=50
##Path = "D://modelsGS//Saving origin_try004_2 soft.tar"  # 정확도 96.10%, root=justresizedM,LR=0.1batch_size=30,epoch=50
##Path = "D://modelsGS//Saving origin_try004_3 soft.tar"  # 정확도 95.93%, root=justresizedM,LR=0.1batch_size=30,epoch=50
##Path = "D://modelsGS//Saving origin_try004_4 soft.tar"  # 정확도 96.45%, root=justresizedM,LR=0.1batch_size=30,epoch=50
Path = "D://modelsGS//Saving origin_try004_5 soft.tar"  # 정확도 %, root=justresizedM,LR=0.1batch_size=30,epoch=50

#                     ----Data 전처리 과정 (root설정,split)-----

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

        #               -원본 data 불러오기(dataset을 Gray Scale로 ImageFolder로 불러온다.)-
tf1=tr.Compose([tr.Grayscale(),tr.ToTensor()])
#tf1=tr.Compose([tr.ToTensor()])

dataset=torchvision.datasets.ImageFolder(root=root,transform=tf1)

        #               -불러온 data split하기(train:0.7, test:0.3*0.5, validation:0.3*0.5)-
train_set,testdata=train_test_split(dataset,train_size=0.7,test_size=0.3,random_state=1004)
test_set,validation_set=train_test_split(testdata,train_size=0.5,test_size=0.5, random_state=1004)

        #               -splited data를 Model에 사용하기 편하게 조절(splited데이터를 Dataloader로)-
trainloader=torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True)
testloader=torch.utils.data.DataLoader(test_set,batch_size=batch_size,shuffle=False)
validationloader=torch.utils.data.DataLoader(validation_set,batch_size=batch_size,shuffle=False)

    #          ----Model 정의(CNN2층,FC2층, Optimizer,criterion)----

# self.FC1=torch.nn.Sequential(nn.Linear(128 * 20 * 25, 128 * 2,bias=True)
#                               ,nn.ReLU())
#
# self.FC2=nn.Linear(128 *2 ,22, bias=False)

#        self.fc1 = nn.Linear(128 * 11 * 8, 128 * 2)        # M1사이즈_(47*57)
#        self.fc1 = nn.Linear(128 * 20 * 25, 128 * 2)        # trsize(G.S Origin 원본94*114일때)

#        self.fc2 = nn.Linear(128 * 2, 22)



#              CNN:합성곱 층, FC:Fully Connected(Linear Layer)
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

    def forward(self, x):
        out = self.CNN1(x)
        out = self.CNN2(out)
        out=torch.flatten(out,1)
        out = self.FC1(out)
        out = self.FC2(out)
        return out



##        return F.softmax(out,dim=1)
##        return F.log_softmax(out, dim=1)

##        return F.softmax(out,dim=1)



model= Net().to(device)
optimizer = optim.Adadelta(model.parameters(),lr=lr)
criterion=nn.CrossEntropyLoss().to(device)
#criterion=nn.NLLLoss().to(device)


#                            --------Train 정의-------
def train(trainloader,model,optimizer,device,epoch,log_interval):
    model.train()
    for batch_idx,(data, label) in enumerate(trainloader):
        data, label = Variable(data).to(device), Variable(label).to(device)
        optimizer.zero_grad()
        output = model(data)
#        print('output1:',output)
        loss=criterion(output,label)
#        print('loss:',loss)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                 epoch, batch_idx * len(data), len(trainloader.dataset),
                 100. * batch_idx / len(trainloader), loss.item()))

    loss_tracker(train_loss_plt, torch.Tensor([loss.item()]), torch.Tensor([epoch]))

#                            --------Test 정의-------
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

#                            --------실제 구동 및 모델 저장-------
for epoch in range(2,epoch+1):
    train(trainloader, model, optimizer, device, epoch,log_interval)
    test(validationloader, model, device, epoch)

##    Path = "D://modelsGS//Saving origin_try000 soft.tar"  # 정확도 92.99%까지, root=justresized1,LR=0.5batch_size=30,epoch=20
##    Path = "D://modelsGS//Saving origin_try001 soft.tar"  # 정확도 94.89%까지,  root=justresized2,LR=0.5batch_size=30,epoch=20
##    Path = "D://modelsGS//Saving origin_try002 soft.tar"  # 정확도 94.89%까지,  root=justresized3,LR=0.5batch_size=30,epoch=20
##    Path = "D://modelsGS//Saving origin_try003 soft.tar"  # 정확도 95.67%까지,  root=justresized4,LR=0.5batch_size=30,epoch=20

##    Path = "D://modelsGS//Saving origin_try004 soft.tar"  # 정확도 95.93%까지, root=justresized_M,LR=0.5batch_size=30,epoch=20
##    Path = "D://modelsGS//Saving origin_try005 soft.tar"  # 정확도 ~~96까지, root=justresized_M,LR=0.5batch_size=30,epoch=20

##    Path = "D://modelsGS//Saving origin_HSVtry000 soft.tar"  # 정확도 95.38%까지, root=HSV_1
##    Path = "D://modelsGS//Saving origin_HSVtry001 soft.tar"  # 정확도 94.51%까지, root=HSV_2
##    Path = "D://modelsGS//Saving origin_HSVtry002 soft.tar"  # 정확도 95.09%까지, root=HSV_3
##    Path = "D://modelsGS//Saving origin_HSVtry003 soft.tar"  # 정확도 96.34%까지, root=HSV_4

##    Path = "D://modelsGS//Saving origin_HSVtry004 soft.tar"  # 정확도 96.34%까지, root=HSV_M
##    Path = "D://modelsGS//Saving origin_HSVtry00_1 soft.tar"  # 정확도 96.34%까지, root=HSV_M
##    Path = "D://modelsGS//Saving origin_try000_1 soft.tar"  # 정확도 93.25%까지, root=justresized1,LR=0.5batch_size=30,epoch=20

    torch.save({
                  'epoch': epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  }, Path)


    ##Path = "D://modelsGS//Saving origin_try000_1 soft.tar"  # 정확도 92.38%, root=justresized1,LR=0.1batch_size=30,epoch=50
    ##Path = "D://modelsGS//Saving origin_try000_2 soft.tar"  # 정확도 93.25%, root=justresized1,LR=0.1batch_size=30,epoch=50
    ##Path = "D://modelsGS//Saving origin_try000_3 soft.tar"  # 정확도 93.33%, root=justresized1,LR=0.1batch_size=30,epoch=50
    ##Path = "D://modelsGS//Saving origin_try000_4 soft.tar"  # 정확도 92.03%, root=justresized1,LR=0.1batch_size=30,epoch=50
    ##Path = "D://modelsGS//Saving origin_try000_5 soft.tar"  # 정확도 93.25%, root=justresized1,LR=0.1batch_size=30,epoch=50
    ##Path = "D://modelsGS//Saving origin_try000_6 soft.tar"  # 정확도 92.03%, root=justresized1,LR=0.1batch_size=30,epoch=50
    ##Path = "D://modelsGS//Saving origin_try000_7 soft.tar"  # 정확도 92.03%, root=justresized1,LR=0.1batch_size=30,epoch=50
    ##Path = "D://modelsGS//Saving origin_try000_8 soft.tar"  # 정확도 92.03%, root=justresized1,LR=0.1batch_size=30,epoch=50
    ##Path = "D://modelsGS//Saving origin_try000_9 soft.tar"  # 정확도 92.03%, root=justresized1,LR=0.1batch_size=30,epoch=50
    ##Path = "D://modelsGS//Saving origin_try000_10 soft.tar"  # 정확도 92.03%, root=justresized1,LR=0.1batch_size=30,epoch=50


##Path = "D://modelsGS//Saving origin_try001_1 soft.tar"  # 정확도 94.89%, root=justresized2,LR=0.1batch_size=30,epoch=50
##Path = "D://modelsGS//Saving origin_try001_2 soft.tar"  # 정확도 94.29%, root=justresized2,LR=0.1batch_size=30,epoch=50
##Path = "D://modelsGS//Saving origin_try001_3 soft.tar"  # 정확도 94.46%, root=justresized2,LR=0.1batch_size=30,epoch=50
##Path = "D://modelsGS//Saving origin_try001_4 soft.tar"  # 정확도 93.59%, root=justresized2,LR=0.1batch_size=30,epoch=50
##Path = "D://modelsGS//Saving origin_try001_5 soft.tar"  # 정확도 94.72%, root=justresized2,LR=0.1batch_size=30,epoch=50
##Path = "D://modelsGS//Saving origin_try001_6 soft.tar"  # 정확도 93.94%, root=justresized2,LR=0.1batch_size=30,epoch=50
##Path = "D://modelsGS//Saving origin_try001_7 soft.tar"  # 정확도 94.29%, root=justresized2,LR=0.1batch_size=30,epoch=50
##Path = "D://modelsGS//Saving origin_try001_8 soft.tar"  # 정확도 93.59%, root=justresized2,LR=0.1batch_size=30,epoch=50
##Path = "D://modelsGS//Saving origin_try001_9 soft.tar"  # 정확도 93.51%, root=justresized2,LR=0.1batch_size=30,epoch=50
##Path = "D://modelsGS//Saving origin_try001_10 soft.tar"  # 정확도 93.77%, root=justresized2,LR=0.1batch_size=30,epoch=50


##Path = "D://modelsGS//Saving origin_try002_1 soft.tar"  # 정확도 94.03%, root=justresized3,LR=0.1batch_size=30,epoch=50
##Path = "D://modelsGS//Saving origin_try002_2 soft.tar"  # 정확도 93.85%, root=justresized3,LR=0.1batch_size=30,epoch=50
##Path = "D://modelsGS//Saving origin_try002_3 soft.tar"  # 정확도 94.37%, root=justresized3,LR=0.1batch_size=30,epoch=50
##Path = "D://modelsGS//Saving origin_try002_4 soft.tar"  # 정확도 93.85%, root=justresized3,LR=0.1batch_size=30,epoch=50
##Path = "D://modelsGS//Saving origin_try002_5 soft.tar"  # 정확도 94.37%, root=justresized3,LR=0.1batch_size=30,epoch=50
##Path = "D://modelsGS//Saving origin_try002_6 soft.tar"  # 정확도 93.77%, root=justresized3,LR=0.1batch_size=30,epoch=50
##Path = "D://modelsGS//Saving origin_try002_7 soft.tar"  # 정확도 94.37%, root=justresized3,LR=0.1batch_size=30,epoch=50
##Path = "D://modelsGS//Saving origin_try002_8 soft.tar"  # 정확도 94.03%, root=justresized3,LR=0.1batch_size=30,epoch=50
##Path = "D://modelsGS//Saving origin_try002_9 soft.tar"  # 정확도 94.37%, root=justresized3,LR=0.1batch_size=30,epoch=50
##Path = "D://modelsGS//Saving origin_try002_10 soft.tar"  # 정확도 93.94%, root=justresized3,LR=0.1batch_size=30,epoch=50

##Path = "D://modelsGS//Saving origin_try003_1 soft.tar"  # 정확도 95.32%, root=justresized4,LR=0.1batch_size=30,epoch=50
##Path = "D://modelsGS//Saving origin_try003_2 soft.tar"  # 정확도 94.20%, root=justresized4,LR=0.1batch_size=30,epoch=50
##Path = "D://modelsGS//Saving origin_try003_3 soft.tar"  # 정확도 94.89%, root=justresized4,LR=0.1batch_size=30,epoch=50
##Path = "D://modelsGS//Saving origin_try003_4 soft.tar"  # 정확도 95.15%, root=justresized4,LR=0.1batch_size=30,epoch=50
##Path = "D://modelsGS//Saving origin_try003_5 soft.tar"  # 정확도 95.15%, root=justresized4,LR=0.1batch_size=30,epoch=50
##Path = "D://modelsGS//Saving origin_try003_6 soft.tar"  # 정확도 95.15%, root=justresized4,LR=0.1batch_size=30,epoch=50
##Path = "D://modelsGS//Saving origin_try003_7 soft.tar"  # 정확도 95.15%, root=justresized4,LR=0.1batch_size=30,epoch=50
##Path = "D://modelsGS//Saving origin_try003_8 soft.tar"  # 정확도 95.5%, root=justresized4,LR=0.1batch_size=30,epoch=50
##Path = "D://modelsGS//Saving origin_try003_9 soft.tar"  # 정확도 94.98%, root=justresized4,LR=0.1batch_size=30,epoch=50
##Path = "D://modelsGS//Saving origin_try003_10 soft.tar"  # 정확도 94.72%, root=justresized4,LR=0.1batch_size=30,epoch=50
