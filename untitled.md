# 图像分类

> 分类问题一直是各领域问题中最容易入门的，mnist数据集前人之述倍矣，不妨试试cifar100，一个常见的图像100分类的数据集

## 一般过程

通常按照 数据集预处理 -> 模型及参数定义 -> 多轮训练 -> 验证模型效果 的步骤进行处理

1. 载入torchvision内置的数据集, 划分训练、验证集。 为了将数据分批载入网络， 通常需要使用DataLoader。

```python
import torch
from torch import optim, nn
import torch.nn.functional as F 
import torchvision
from torchvision import datasets, models, transforms

device = torch.device("cuda:0")
batch_size = 20
epochs = 100
log_interval = 100
img_size = 280
# transforms
transform = transforms.Compose([    
                transforms.Resize(img_size),
                transforms.RandomHorizontalFlip(p=0.5),
                #transforms.RandomVerticalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([ 0.485,0.456,0.406 ],[ 0.229,0.224,0.225 ])
                ]) 
# dataset
trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)
   
```

2\. 使用预置网络resnet18， 改变分类层的输出维度--等于数据集的label数量

```python
# !pip install efficientnet_pytorch
num_classes = 100
# 使用预置的残差网络
def net_res():
  net = models.resnet18(pretrained=True)
  net.fc = nn.Linear(net.fc.in_features, num_classes )
  return net
# 定义优化器和损失函数
optimizer = torch.optim.SGD(net.parameters(), 
                lr=1e-3, 
                momentum=0.9
                  )
schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5, last_epoch=-1)
loss = nn.CrossEntropyLoss()  
criterion = F.cross_entropy
```

3\. 定义训练、测试函数。然后开始训练

```python
def train(model, device, train_loader):
    model.train()
    running_loss = 0.0
    for i,data in enumerate(train_loader):
        inputs ,labels = data[0].to(device) , data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        # update weights
        optimizer.step()
        running_loss += loss.item()
        if i % log_interval == log_interval - 1:
          print('Training: {} loss: {}'.format(i + 1, running_loss / log_interval))
          running_loss = 0.0
            
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Test : Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

# 开始训练，并保存网络的权重。
for epoch in range(epochs):
    print("EPOCHS：{}/{}".format(epoch+1,epochs)) 
    
    train(net,device,train_loader)
    test(net,device,test_loader)
    #schedule.step()
    torch.save(net.state_dict(), "model_weights_e{}".format(epoch+1) )
print("Finished Training")
```

4\. 载入网络和权重， 对任意图片进行类别预测。

还没写的，网上一大堆现成的。
