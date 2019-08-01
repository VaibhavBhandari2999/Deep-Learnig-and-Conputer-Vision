
#This si being run on the same 'virtual_platform' virtual environment which we made previuosly through anaconda prompt.

#Real images will be from CIFAR10 dataset

# Importing the libraries
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

# Setting some hyperparameters
batchSize = 64 # We set the size of the batch.
imageSize = 64 # We set the size of the generated images (64x64).


# Creating the transformations. The transformations are only for Generator Neural Network. This manipulates the image to make it compatible with the Neural Network
transform = transforms.Compose([transforms.Scale(imageSize), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]) # We create a list of transformations (scaling, tensor conversion, normalization) to apply to the input images. ToTenser() converts a PIL Image or numpy.ndarray to tensor. Normalize basically Normalize a tensor image with mean and standard deviation.  

# Loading the dataset. The dataset is downloaded and put in that directory.
dataset = dset.CIFAR10(root = './data', download = True, transform = transform) # We download the training set in the ./data folder and we apply the previous transformations on each image.
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchSize, shuffle = True, num_workers = 2) # We use dataLoader to get the images of the training set batch by batch.Shuffle basically shuffles the images and reorders them. Workers=2 means two threads are used to execute. This make the whole process faster.

# Defining the weights_init function that takes as input a neural network m and that will initialize all its weights.
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:                #In the classname which calls the function, 'Conv' is searched for in the init function of the Generator NN. If it is found, it specifies the wieghts. In the G class, Conv is found in ConvTranspose2d
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:         #In the classname which calls the function, , 'BatchNorm' is searched for and the weights and bias put respectively. In the G class, BatchNorm is found in BatchNorm2d
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# Defining the generator

class G(nn.Module): # We introduce a class to define the generator. We apply inheritance and inherit from nn.module
    #Basically, init(initialization) function basically defines all the properties of the object of the class
    def __init__(self): # We introduce the __init__() function that will define the architecture of the generator.
        super(G, self).__init__() # We inherit from the nn.Module tools.
        
        #The architecture of the NN is got from experimentation
        self.main = nn.Sequential( # We create a meta module of a neural network that will contain a sequence of modules (convolutions, full connections, etc.).
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias = False), # We start with an inversed convolution. In a normal convolutional network, input is a photo and output is a signal/noise/vector. But we need noise/signal/vector as input and photo as output. That is why we use inverse convolutional network. The first arguement is 100(the inputof our inverse CNN will be a vector of size 100), 2nd arguement is number of feature maps of the output, 3rd arguement is size of kernal(will be of size 4x4), 4th arguement is stride(which controls stride for cross validation), 5th arguement is 0 which is the padding, 6th arguement is bias(which we dont want here, so we set it as False). Get detaisl of all these by selecting 'ConvTranspose2d(' nd pressing Cntrl+I. For why bias is false, see https://stackoverflow.com/questions/2480650/role-of-bias-in-neural-networks
            nn.BatchNorm2d(512), # We normalize all the features along the dimension of the batch. We have 512 featuremaps and we batchnorm each of them 
            nn.ReLU(True), # We apply a ReLU rectification to break the linearity.
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False), # We add another inversed convolution same as before except input of this is output of previous convolution layer which was 512 and new number of outputs is 256(got by experinmentation), kernal size of 2 and padding of 1
            nn.BatchNorm2d(256), # We normalize again. Now as we have 256 feature maps as output of 2nd layer, we have to batchnorm all of them
            nn.ReLU(True), # We apply another ReLU.
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False), # We add another inversed convolution. Same as 2nd layer except input is 256(output of previous layer) and output is 128
            nn.BatchNorm2d(128), # We normalize again.
            nn.ReLU(True), # We apply another ReLU.
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False), # We add another inversed convolution.Same as 2nd layer except input is 128(output of previous layer) and output is 64
            nn.BatchNorm2d(64), # We normalize again.
            nn.ReLU(True), # We apply another ReLU.
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias = False), # We add another inversed convolution. Same as 2nd layer except input is 64(output of previous layer) and output is 3
            nn.Tanh() # We apply a Tanh rectification to break the linearity and stay between -1 and +1.
        )
        
        #See https://datascience.stackexchange.com/questions/6107/what-are-deconvolutional-layers
        
        
    def forward(self, input): # We define the forward function that takes as argument an input that will be fed to the neural network(we take 100 as input in init function in inverse convolutional layer), and that will return the output containing the generated images.
        output = self.main(input) # We forward propagate the signal through the whole neural network of the generator defined by self.main.
        return output # We return the output containing the generated images.
    
# Creating the generator
netG = G() # We create the generator object.
netG.apply(weights_init) # We initialize all the weights of its neural network. Apply function is used

# Defining the discriminator

class D(nn.Module): # We introduce a class to define the discriminator.
    
        #All the explanations are similar to generator class unless sepcified otherwise
    def __init__(self): # We introduce the __init__() function that will define the architecture of the discriminator.
        super(D, self).__init__() # We inherit from the nn.Module tools.
        self.main = nn.Sequential( # We create a meta module of a neural network that will contain a sequence of modules (convolutions, full connections, etc.).
            nn.Conv2d(3, 64, 4, 2, 1, bias = False), # We start with a convolution. As in genorator we needed vector as input and image as output, now we need viceversa as the image coming from generator is input and a value is output. So we use normal convolutional layer, not inverse. Also, input of this will be output of generator(which was 3) and output wil be 64. Everything else is same as generator
            nn.LeakyReLU(0.2, inplace = True), # We apply a LeakyReLU. Search leaky relu on google and compare graph of it with ReLU graph. See pytorch documentation for leakyReLU 
            nn.Conv2d(64, 128, 4, 2, 1, bias = False), # We add another convolution. INput will be output of previous convolutional layer
            nn.BatchNorm2d(128), # We normalize all the features along the dimension of the batch.
            nn.LeakyReLU(0.2, inplace = True), # We apply another LeakyReLU.
            nn.Conv2d(128, 256, 4, 2, 1, bias = False), # We add another convolution.
            nn.BatchNorm2d(256), # We normalize again.
            nn.LeakyReLU(0.2, inplace = True), # We apply another LeakyReLU.
            nn.Conv2d(256, 512, 4, 2, 1, bias = False), # We add another convolution.
            nn.BatchNorm2d(512), # We normalize again.
            nn.LeakyReLU(0.2, inplace = True), # We apply another LeakyReLU.
            nn.Conv2d(512, 1, 4, 1, 0, bias = False), # We add another convolution. Here the final output should be only 1 as we sam in the discriminator logic. We change stride and padding also
            nn.Sigmoid() # We apply a Sigmoid rectification to break the linearity and stay between 0 and 1.We only require 0 and 1 as the focus of a discriminator is that 0 corresponds to rejection of image and 1 accepts the image, and the values between 0 and 1 indicate to what percentage the image is accepted or rejected
        )
        
    
    def forward(self, input): # We define the forward function that takes as argument an input that will be fed to the neural network, and that will return the output which will be a value between 0 and 1.
        output = self.main(input) # We forward propagate the signal through the whole neural network of the discriminator defined by self.main.
        return output.view(-1) # We return the output which will be a value between 0 and 1. At the end of a normal CNN we flatten the result of all the convolutions, here too we must flatten the result to make sure all elements of outptu are along one same dimension. This dimension corresponds to dimension of batch size, in pytorch we use view function
    
# Creating the discriminator
netD = D() # We create the discriminator object.
netD.apply(weights_init) # We initialize all the weights of its neural network.
        

#Training is broken down into 2 big steps, update the weights in the neural network of discriminator and then updating the wights of NN of generator
#While first training the discriminator we will have to train it to see whats real and whats fake. So we will first train it by giving it a real image and set the traget to 1(which means image is accepted) and then another trainign by giving it a fake image and setting target to 0(image is not accepted). Now to train the weights, we'll take the fake image again, then feed this into discrimator to get output(which is bw 0 and 1). Then we compare the loss between output of discrimator(between 0 and 1) and 1.We'll bck propogate this error into NN of generator , the we appy stochastic gradient descent


criterion = nn.BCELoss() # We create a criterion object that will measure the error between the prediction and the target. reates a criterion that measures the Binary Cross Entropy between the target and the output:
optimizerD = optim.Adam(netD.parameters(), lr = 0.0002, betas = (0.5, 0.999)) # We create the optimizer object of the discriminator. Learning Rate is taken as 0.0002. Search what betas is
optimizerG = optim.Adam(netG.parameters(), lr = 0.0002, betas = (0.5, 0.999)) # We create the optimizer object of the generator.

for epoch in range(25): # We iterate over 25 epochs. We can take more or less, but 25 gives a good value

    for i, data in enumerate(dataloader, 0): # We iterate over the images of the dataset in each epoch. dataloader gives us the mini batches, and 0 defines which index the loops starts from
         # So, 1st step Updating the weights of the neural network of the discriminator
         # Training the discriminator with a real image of the dataset
         # Training the discriminator with a fake image generated by the generator
         # 2nd Step: Updating the weights of the neural network of the generator
         # 3rd Step: Printing the losses and saving the real images and the generated images of the minibatch every 100 steps
          
              
          
        # 1st Step: Updating the weights of the neural network of the discriminator        

        netD.zero_grad() # We initialize to 0 the gradients of the discriminator with respect to the weights.
        
        real, _ = data # We get a real image of the dataset which will be used to train the discriminator. The data in each minibatch is brought. But this gives 2 elements, we only care about first one, thats why 2nd arguement is taken as underscore
        input = Variable(real) # We wrap it in a variable because the above is not yet an accepted input in pytorch standards which accept only inputs in torch variables(which are highly advanced variable which contains tensor and gradient). So now our input images are not only in mini batch but also in torch variable
        target = Variable(torch.ones(input.size()[0])) # We get the target.input.size()[0] will return size minibatch. So we add variable class in parentheses as target becomes torch variable. torch.ones is used to create the input tensor
        output = netD(input) # We forward propagate this real image into the neural network of the discriminator to get the prediction (a value between 0 and 1).
        errD_real = criterion(output, target) # We compute the loss between the predictions (output) and the target (equal to 1).
        