"""Model class template

This module provides a template for users to implement custom models.
You can specify '--model template' to use this model.
The class name should be consistent with both the filename and its model option.
The filename should be <model>_dataset.py
The class name should be <Model>Dataset.py

You need to implement the following functions:
    <modify_commandline_options>:　Add model-specific options and rewrite default values for existing options.
    <__init__>: Initialize this model class.
    <set_input>: Unpack input data and perform data pre-processing.
    <forward>: Run forward pass. This will be called by both <optimize_parameters> and <test>.
    <optimize_parameters>: Update network weights; it will be called in every training iteration.
"""
import torch
from .base_model import BaseModel


class TemplateModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel

        # specify the training losses you want to print out. The program will call base_model.get_current_losses to plot the losses to the console and save them to the disk.
        self.loss_names = ['filler', 'accuracy']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks to save and load networks.
        self.model_names = ['_filler']

        # define networks, random filler network...
        self.net_filler = torch.nn.Linear(in_features=10, out_features=1, bias=False)
        self.net_filler = self.net_filler.to(self.device)

        # you can use opt.isTrain to specify different behaviors for training and test. 
        if self.isTrain: 
            # ENTER HERE: You can inialize your loss, optimizer and scheduler here 
            self.criterionLoss = None
            self.optimizer = None
            self.optimizers = [self.optimizer]

        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

    def train(self):
        self.net_filler.train()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: whatever you return from your dataloader! Up to you. 
            Set class member variables to use in forward etc.
        """
        print('*********************************\nThis is a empty template, please see example versions !!! \n*********************************')
        raise NotImplementedError

    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        # ENTER HERE: pass self.input through network
        print('*********************************\nThis is a empty template, please see example versions !!! \n*********************************')
        raise NotImplementedError

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results

        # ENTER HERE: pass self.input through network
        print('*********************************\nThis is a empty template, please see example versions !!! \n*********************************')
        raise NotImplementedError

    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        # ENTER HERE: Uncomment optimizer lines when it is initialized
        # self.optimizer.zero_grad()   # clear network's existing gradients 
        self.forward()               # first call forward to calculate results
        self.backward()              # calculate gradients for network
        # self.optimizer.step()        # update gradients for network 
