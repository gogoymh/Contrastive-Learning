import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import network

class ContrastiveSupervisionLoss(nn.Module):
    def __init__(self, num_class, device):
        super().__init__()
        
        self.num_class = num_class
        self.one_hot_label = self._make_one_hot_vector(num_class)
        self.one_hot_label = self.one_hot_label.to(device)
        self.softmax = nn.Softmax(dim=1)
        self.device = device
        
    def _make_one_hot_vector(self, num_class):
        vectors = torch.zeros((num_class, num_class)).float()
        for i in range(num_class):
            vectors[i,i] = 1.0
        
        return vectors
        
    def L1_distance(self, a, b):
        return torch.abs(a - b).sum(dim=-1)
    
    def _make_matrix(self, a, b):
        matrix = self.L1_distance(a.unsqueeze(1), b.unsqueeze(0))
        return matrix
    
    def _prepare_data_mask(self, y):
        #print(y)
        unique = y.unique()
        #print(unique)
        
        cat = torch.cat((self.one_hot_label[y], self.one_hot_label[unique]), dim=0)
        mask = 1 - self._make_matrix(cat, cat)/2 - torch.eye(unique.shape[0]+3).to(self.device)
        #print(mask)
        
        return mask.to(self.device)
        
    def forward(self, output, y, mask):
        unique = y.unique()
        softmax = self.softmax(output)
        cat = torch.cat((softmax, self.one_hot_label[unique]), dim=0)
        
        distance_matrix = self._make_matrix(cat, cat)
        print(distance_matrix)
        
        distance = distance_matrix * mask
        print(distance)
        
        return distance.sum()/2

device = torch.device("cuda:0")
criterion = ContrastiveSupervisionLoss(4, device) #nn.CrossEntropyLoss()

x = torch.Tensor([[-1,3,-0.5,-2],[1,2,3,4],[4,-1,3,2]]).float().to(device)
y = torch.Tensor([1,2,3]).long().to(device)

mask = criterion._prepare_data_mask(y)
print(mask)
loss = criterion(x, y, mask)
print(loss)

'''
train_loader = DataLoader(
                datasets.CIFAR10(
                        "./data/CIFAR10",
                        train=True,
                        download=True,
                        transform=transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
                                ),
                        ),
                batch_size=3, shuffle=True)#, pin_memory=True)


test_loader = DataLoader(
                datasets.CIFAR10(
                        './data/CIFAR10',
                        train=False,
                        download=True,
                        transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(
                                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
                                ),
                        ),
                batch_size=64, shuffle=False)#, pin_memory=True)

device = torch.device("cuda:0")
model = network.resnet56(num_classes=10).to(device)

#optimizer = optim.Adam(model.parameters(), lr=0.1)
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,150], gamma=0.1)
criterion = ContrastiveSupervisionLoss(10, device) #nn.CrossEntropyLoss()

batch_size = 128
batch_num = len(train_loader.dataset)//batch_size
print(batch_num)

best_acc = 0
for epoch in range(300): # 원하는 만큼
    runnning_loss = 0
    
    for i in range(batch_num):
        optimizer.zero_grad()
        loss = 0
        for j in range(batch_size//3):
            x, y = train_loader.__iter__().next()
            x = x.float().to(device)
            y = y.long().to(device)
            mask = criterion._prepare_data_mask(y)
            output = model(x)
            loss += criterion(output, y, mask)
        loss.backward()
        optimizer.step()
        runnning_loss += loss.item()
        print("[%d/%d] [Loss:%f]" % (i+1, batch_num, loss.item()))
    
    runnning_loss /= batch_num
    print("[Epoch:%d] [Loss:%f]" % ((epoch+1), runnning_loss), end=" ")
    
    accuracy = 0
    with torch.no_grad():
        model.eval()
        correct = 0
        for x, y in test_loader:
            output = model(x.float().to(device))
            pred = output.argmax(1, keepdim=True)
            correct += pred.eq(y.long().to(device).view_as(pred)).sum().item()
                
        accuracy = correct / len(test_loader.dataset)
        
        if save:
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': losses[epoch].item()}, "/home/cscoi/MH/resnet56_real4.pth")
            save_again = False
            print("[Accuracy:%f]" % accuracy)
            print("Saved early")
            break
        
        if accuracy >= best_acc:
            print("[Accuracy:%f] **Best**" % accuracy)
            best_acc = accuracy
        else:
            print("[Accuracy:%f]" % accuracy)
        model.train()
        
    scheduler.step()
'''




















