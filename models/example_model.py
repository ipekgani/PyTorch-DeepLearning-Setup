"""
    Example that fills the template with EfficientNet
    You can follow this example to replace everthing with your model and losses etc.
"""

import torch, os
from torch import nn
from .base_model import BaseModel
from . import get_optimizer, get_scheduler
from efficientnet_pytorch import EfficientNet
from torchsummary import summary

class ExampleModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--eff_model', type=str, default='b0')
        parser.add_argument('--pretrained_eff', action='store_true')

        return parser

    def __init__(self, opt):
        """
            example model initializer for EfficientNet
        """
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel

        # name of the loss metrics for logging easily. Write the extension after variables: loss_* 
        self.loss_names = ['ce', 'accuracy']
        self.model_names = ['_eff']
        
        # get model from pytorch (or define your own model)
        model = EfficientNet.from_pretrained('efficientnet-%s' % opt.eff_model)
        # replace fc layer wrt number of classes in task
        model._fc = torch.nn.Linear(in_features=model._fc.in_features, out_features=opt.num_classes)

        self.net_eff = model
        self.net_eff.to(self.device)

        # define loss
        self.criterionLoss = torch.nn.CrossEntropyLoss()

        # define optimizers and schedulers
        if self.isTrain:
            # created in lists in case you need multiple optimizers for different networks. (e.g. original pix2pix optimized two networks because its GANs)
            self.optimizer = get_optimizer(opt, self.net_eff.parameters())
            self.optimizers = [self.optimizer]
            self.schedulers = [get_scheduler(self.optimizer, opt)]

        # program will automatically call <model.setup> to load networks, and print networks

    def train(self):
        self.net_eff.train()

    def cpu(self):
        self.net_eff = self.net_eff.cpu()

    def eval(self):
        self.net_eff.eval()

    def get_net(self):
        return self.net_eff

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains e data itself and its metadata information.
        """
        batch, label, *_ = input
        self.input = batch.to(self.device)
        self.ground_truth = label.to(self.device) 

    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        
        self.output = self.net_eff(self.input)

    def test(self):
        """Forward function used in test time.
        """
        with torch.no_grad():
            self.forward()
            self.output_softmax = torch.softmax(self.output, dim=-1)                            # softmax over number of classes
            self.preds = self.output_softmax.argmax(dim=-1)                                     # get class predictions

            if self.isTrain:                                                                    #* would only be entered in validation
                self.loss_ce = self.criterionLoss(self.output, self.ground_truth.long())        # compute loss
                self.loss_accuracy = (self.ground_truth.eq(self.preds)).float().mean().item()   # compute accuracy (for logging)
            else:
                self.loss_accuracy, self.loss_ce = None, None

    def get_output(self): 
        return self.output_softmax, self.output, self.preds, self.loss_ce                       # return metrics for logging

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results
        self.output_softmax = torch.softmax(self.output, dim=-1)                        # softmax over number of classes
        self.preds = self.output_softmax.argmax(dim=-1)                                 # get class predictions
        self.loss_ce = self.criterionLoss(self.output, self.ground_truth.long())        # compute loss
        self.loss_ce.backward()                                                         # backward loss
        self.loss_accuracy = (self.ground_truth.eq(self.preds)).float().mean().item()   # compute accuracy (for logging)

    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.optimizer.zero_grad()   # clear network's existing gradients 
        self.forward()               # first call forward to calculate results
        self.backward()              # calculate gradients for network
        self.optimizer.step()        # update gradients for network 

