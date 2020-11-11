import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import h5py
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F


train_set_img = []
train_set_label = []

dataset_path = '/home/rakumar/char_segmentation/trainingDataset.h5'

train_model = True
save_model = True
load_saved_model = True

mapping=['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

def load_dataset():
    global train_set_img, train_set_label

    train_dataset = h5py.File(dataset_path, 'r')         # test set
    train_set_img = np.array(train_dataset['image'][:]) # train set features
    train_set_label = np.array(train_dataset['label'][:]) # train set labels

    train_set_img = torch.FloatTensor(train_set_img)
    train_set_label = train_set_label.astype(np.int)
    train_set_label = torch.LongTensor(train_set_label)

    # plot img
    rows = 3
    cols = 3
    axes=[]
    fig=plt.figure()

    i =0
    for a in range(rows*cols):
        b = train_set_img[i]
        axes.append( fig.add_subplot(rows, cols, a+1) )

        subplot_title=('class- '+mapping[train_set_label[i]])
        axes[-1].set_title(subplot_title)
         
        i+=1
        b = b.squeeze(0)
        b = b.numpy() # pyTorch tensor to numpy
        plt.imshow(b.astype('uint8'), interpolation='nearest', aspect=15)
    fig.tight_layout()    
    plt.show()

load_dataset()


# Define a CNN

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)

        self.fc1 = nn.Linear(16 * 1 * 7, 100)
        self.fc2 = nn.Linear(100, 84)
        self.fc3 = nn.Linear(84, 36)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 1 * 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
print(net)


### Define a Loss function and optimizer ###
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr = 0.0001)


### Train the network ###
def modelTraining():
  loss_values=[]
  for epoch in range(10):  # loop over the dataset multiple times
      running_loss = 0.0
      for i in range(len(train_set_label)):
          # get the inputs
          inputs = train_set_img[i]
          #inputs /= 255.0
          inputs = inputs.unsqueeze(0)
          inputs = inputs.unsqueeze(0)

          labels = train_set_label[i]
          labels = torch.tensor([labels])

          # zero the parameter gradients
          optimizer.zero_grad()

          # forward + backward + optimize
          outputs = net(inputs)   # forward

          loss = criterion(outputs, labels)
          loss.backward()                             # backward
          optimizer.step()                            # optimize

          # print statistics
          running_loss += loss.item()
          if i % 100 == 99:    # print every 100 mini-batches
              print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 100))
              loss_values.append(running_loss)
              running_loss = 0.0
  plt.plot(loss_values)
  plt.show()        
        
  print('Finished Training')

if train_model:
  modelTraining()
  
# save trained model:
PATH = '/home/rakumar/char_segmentation/model.pth'

if save_model:
  torch.save(net.state_dict(), PATH)

# load back the saved model
if load_saved_model:
  net = Net()
  net.load_state_dict(torch.load(PATH))

# =========================================================================================

## TEST on only few/single Dataset
test_images = [train_set_img[0], train_set_img[1], train_set_img[2]]
labels = [train_set_label[0],train_set_label[1],train_set_label[2]]
for i, image in enumerate(test_images):
    test_image = image
    test_image = test_image.unsqueeze(0)
    test_image = test_image.unsqueeze(0)

    outputs = net(test_image)
    _, predicted = torch.max(outputs, 1)
    
    # print('GroundTruth: ', chr(decode_to_labels(torch.tensor(labels[i].numpy()))) )
    # print('Predicted: ',  chr(decode_to_labels(torch.tensor(predicted.numpy()[0]))) )
    print('GroundTruth: ', mapping[labels[i].numpy()])
    print('Predicted: ', mapping[predicted.numpy()[0]])


# network performance on the whole dataset
correct = 0
total = 0
with torch.no_grad():
    for i in range(len(train_set_label)):
        images = train_set_img[i]
        images = images.unsqueeze(0)
        images = images.unsqueeze(0)

        labels = train_set_label[i]
        labels = torch.tensor([labels])

        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the '+str(len(train_set_label))+' test images (%d/%d) : %.2f %%' % (correct, total ,100 * correct / total))


# accuracy for all classes:
class_correct = list(0. for i in range(36))
class_total = list(0. for i in range(36))
with torch.no_grad():
  for i in range(len(train_set_label)):
        images = train_set_img[i]
        images = images.unsqueeze(0)
        images = images.unsqueeze(0)

        labels = train_set_label[i]
        labels = torch. tensor([labels])

        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        
        c = (predicted == labels).squeeze()
        
        class_correct[labels] += c.item()
        class_total[labels] += 1
        
for i in range(36):
    # print('Accuracy of %5s  (%d/%d) : %.2f %%' % (chr(decode_to_labels(torch.tensor(i))), class_correct[i], class_total[i] ,100 * class_correct[i] / class_total[i]))
    print('Accuracy of %5s  (%d/%d) : %.2f %%' % (mapping[i], class_correct[i], class_total[i] ,100 * class_correct[i] / class_total[i]))

# =========================================================================================
