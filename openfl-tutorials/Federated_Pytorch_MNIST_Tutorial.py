#!/usr/bin/env python
# coding: utf-8

# # Federated PyTorch MNIST Tutorial
#Install dependencies if not already installed
#get_ipython().system('pip install torch torchvision')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
import openfl.native as fx
from openfl.federated import FederatedModel,FederatedDataSet


# After importing the required packages, the next step is setting up our openfl workspace. To do this, simply run the `fx.init()` command as follows:

# In[ ]:


#Setup default workspace, logging, etc.
fx.init('torch_cnn_mnist', log_level='METRIC', log_file='./spam_metric.log')


# Now we are ready to define our dataset and model to perform federated learning on. The dataset should be composed of a numpy arrayWe start with a simple fully connected model that is trained on the MNIST dataset. 

# In[ ]:


def one_hot(labels, classes):
    return np.eye(classes)[labels]

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
train_images,train_labels = trainset.train_data, np.array(trainset.train_labels)
train_images = torch.from_numpy(np.expand_dims(train_images, axis=1)).float()

validset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)

valid_images,valid_labels = validset.test_data, np.array(validset.test_labels)
valid_images = torch.from_numpy(np.expand_dims(valid_images, axis=1)).float()
valid_labels = one_hot(valid_labels,10)

# In[ ]:


feature_shape = train_images.shape[1]
classes       = 10

fl_data = FederatedDataSet(train_images,train_labels,valid_images,valid_labels,batch_size=32,num_classes=classes)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(32 * 5 * 5, 32)
        self.fc2 = nn.Linear(32, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class Net2_L5(nn.Module):
    def __init__(self):
        super(Net2_L5, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(1, 32, 2),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 2),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 2),
            nn.ReLU(),
        )
        self.fc1 = nn.Linear(128 * 5**2 , 200)
        self.fc2 = nn.Linear(200, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.block(x)
        out = out.view(-1,  128 * 5**2)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return F.log_softmax(out,1)
    
optimizer = lambda x: optim.Adam(x, lr=1e-4)

def cross_entropy(output, target):
    """Binary cross-entropy metric
    """
    return F.cross_entropy(input=output,target=target)


# Here we can define metric logging function. It should has the following signature described below. You can use it to write metrics to tensorboard or some another specific logging.

# In[ ]:


from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('./logs/cnn_mnist', flush_secs=5)


def write_metric(node_name, task_name, metric_name, metric, round_number):
    writer.add_scalar("{}/{}/{}".format(node_name, task_name, metric_name),
                      metric, round_number)


# In[ ]:



#Create a federated model using the pytorch class, lambda optimizer function, and loss function
fl_model = FederatedModel(build_model=Net,optimizer=optimizer,loss_fn=cross_entropy,data_loader=fl_data)


# The `FederatedModel` object is a wrapper around your Keras, Tensorflow or PyTorch model that makes it compatible with openfl. It provides built in federated training and validation functions that we will see used below. Using it's `setup` function, collaborator models and datasets can be automatically defined for the experiment. 

# In[ ]:


#collaborator_models = fl_model.setup(num_collaborators=2)
#collaborators = {'one':collaborator_models[0],'two':collaborator_models[1]}#, 'three':collaborator_models[2]}
collaborator_models = fl_model.setup(num_collaborators=5)
collaborators = {'one':collaborator_models[0], 'two':collaborator_models[1],
                'three':collaborator_models[2], 'four':collaborator_models[3],
                'five':collaborator_models[4]}

# In[ ]:


#Original MNIST dataset
print(f'Original training data size: {len(train_images)}')
print(f'Original validation data size: {len(valid_images)}\n')

#Collaborator one's data
print(f'Collaborator one\'s training data size: {len(collaborator_models[0].data_loader.X_train)}')
print(f'Collaborator one\'s validation data size: {len(collaborator_models[0].data_loader.X_valid)}\n')

#Collaborator two's data
print(f'Collaborator two\'s training data size: {len(collaborator_models[1].data_loader.X_train)}')
print(f'Collaborator two\'s validation data size: {len(collaborator_models[1].data_loader.X_valid)}\n')

#Collaborator three's data
print(f'Collaborator three\'s training data size: {len(collaborator_models[2].data_loader.X_train)}')
print(f'Collaborator three\'s validation data size: {len(collaborator_models[2].data_loader.X_valid)}')


# We can see the current plan values by running the `fx.get_plan()` function

# In[ ]:


#Get the current values of the plan. Each of these can be overridden
print(fx.get_plan())


# Now we are ready to run our experiment. If we want to pass in custom plan settings, we can easily do that with the `override_config` parameter

# In[ ]:


# Run experiment, return trained FederatedModel

final_fl_model = fx.run_experiment(collaborators, override_config={
    'aggregator.settings.rounds_to_train': 10,
    'aggregator.settings.log_metric_callback': write_metric,
    'secret_collaborator.settings.watermark': 'false'
})


# In[ ]:


#Save final model
final_fl_model.save_native('final_pytorch_model')

