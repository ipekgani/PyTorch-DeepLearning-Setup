# PyTorch-DeepLearning-Setup

An easy-to-use PyTorch deep learning framework adapted from [Pix2Pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). With this as basis you can easily implement your models and train them right away. All training, validation, saving models etc. is already provided. 

I liked Pix2Pix setup a lot and wanted to provide my version of it for others. The setup is very easy to learn and use. New models and dataloaders can be added very easily by filling template classes. Other than that there is a single model and data agnostic training file that can be used to run everything. 

You can run an example as follows below. It will use [EfficientNet](https://github.com/lukemelas/EfficientNet-PyTorch) and CIFAR-10 data as implemented in example files.
```python
python train.py --dataroot datasets --dataset_mode example --model example --validation_datamode example --validation --num_threads 0 --batch_size 16 --num_classes 10 
```

## Soon

I will add an augmentation datalaoder and possibly also logging stuff.