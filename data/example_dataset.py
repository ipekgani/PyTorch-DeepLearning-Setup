"""
    Example that fills the template with CIFAR10
    You can follow this example to replace everthing with your data loading
"""
from data.base_dataset import BaseDataset
import torchvision.datasets as datasets
from torchvision import transforms
import torch

class ExampleDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """

        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)

        # basic loading transformations
        transform = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor()])

        # get your dataset. Normally here you would load image paths into a list
        self.cifar_10 = datasets.CIFAR10(root=opt.dataroot, train=opt.isTrain, download=True, transform=transform)

        # define your own transforms... normalization etc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

           Example data loading using MNIST. Return torch tensors of data and labels etc.
        """
        # index your image path list. Then load image 
        image_tensor, label_tensor = self.cifar_10.__getitem__(index)
        # you could additionally normalize your data here if necessary
        
        return image_tensor, label_tensor

    def __len__(self):
        """Return the total number of images."""
        return 1000 # len(self.cifar_10) normally return dataset length
