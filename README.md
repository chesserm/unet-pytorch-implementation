# U-Net Implementation in PyTorch 

This repo contains an implementation of the U-Net architecture from [the paper](https://arxiv.org/abs/1505.04597).

The model is defined in `unet.py` with sample training code in `train.py`. Note that the code within `train.py` is not complete. There are clear `# TODO` comments defined for where things need to change to be functional (e.g., define your dataset, define your loss functions, log your metrics, etc.) and the code is otherwise well commented.

Within `unet.py` you have the following classes:
- `UnetModel`: The actual U-Net model, complete with network heads. This is what you will be instantiating and modifying as needed (e.g., if you want to change the network heads or bottleneck sizes)
- `UnetEncode`: A class to represent the encoder (downsampling) side of the U-Net architecture. This class is composed of `DownBlock` block objects.
- `UnetDecode`: A class to represent the decoder (upsampling) side of the U-Net architecture. This class is composed of `UpBlock` block objects.
- `DownBlock`: This class represents re-usable network blocks for the encoder portion of the Unet architecture.
- `UpBlock`: This class represents re-usable network blocks for the decoder portion of the Unet architecture.
- `CoreBlock`: This is a fundamental building block of architecture. It is composed of a 2D convolutional layer, a normalization, and `ReLU()` activation function. 

## Some important notes:
- The bottleneck (portion of UNet connecting between encoder and decoder halves of the architecture) is defined in the `UnetModel` constructor.
- The network heads are defined in the constructor of the `UnetModel`. This code was originally created for instance segmentation for self-driving, so there is a mask head and depth regression head. Change these as needed and **be sure to adjust the forward function and loss functions accordingly**. 
- The `skip_between_blocks` and `skip_within_blocks` flags should be set to false for now. In translating this to a standalone repo, some bugs were introduced that need to be addressed.

Additionally, note that this code was adapted from code created for a project in the deep learning course I took as part of my Master's degree at the University of Texas at Austin.
