from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class Classifier(nn.Module):
    def __init__(
        self,
        l0_channels: int = 64,
        in_channels: int = 3,
        num_classes: int = 6,
        num_blocks: int=2
    ):
        """
        A convolutional network for image classification.

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()
        
        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))
        
        self.l0_channels = l0_channels
        self.num_blocks = num_blocks

        class Block(nn.Module):
            """
            Block for part of CNN. Currently setup to skip 2 subblocks (each subblock is Conv, norm, activation)
            """
            def __init__(self, in_channels, out_channels, stride=2):
                super().__init__()

                kernel_size = 3
                padding = (kernel_size - 1) // 2

                self.model = nn.Sequential(
                    nn.Conv2d(in_channels,  out_channels, kernel_size, stride, padding),
                    # nn.GroupNorm(1, out_channels),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding),
                    # nn.GroupNorm(1, out_channels),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU()
                )
                self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0) if in_channels != out_channels else nn.Identity()
                return 
            
            def forward(self, x):
                return self.skip(x) + self.model(x)



        # Initial layer is special
        layers = [nn.Conv2d(in_channels, out_channels=l0_channels, kernel_size=7, stride=2, padding=3),
                  nn.ReLU()
                  ]

        input_channels = l0_channels
        for _ in range(num_blocks):
            # Because we are striding by 2
            output_channels = input_channels * 2
            layers.append(Block(input_channels, output_channels, stride=2))
            input_channels = output_channels

        # Setup final layer to be the 1x1 convolution
        layers.append(nn.Conv2d(input_channels, num_classes, kernel_size=1))
        
        self.model = nn.Sequential(*layers)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, h, w) image

        Returns:
            tensor (b, num_classes) logits
        """
        # optional: normalizes the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        return self.model(z).mean(dim= -1).mean(dim=-1)


    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Used for inference, returns class labels
        This is what the AccuracyMetric uses as input (this is what the grader will use!).
        You should not have to modify this function.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            pred (torch.LongTensor): class labels {0, 1, ..., 5} with shape (b, h, w)
        """
        return self(x).argmax(dim=1)


class Detector(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
        initial_output_channels:int=64,
        num_blocks:int=2,
        num_convs_per_block:int=2,
        skip_within_blocks:bool=False,
        skip_between_blocks:bool=False
    ):
        """
        A single model that performs segmentation and depth regression

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.num_convs_per_block = num_convs_per_block
        self.skip_within_blocks = skip_within_blocks
        self.skip_between_blocks = skip_between_blocks

        class CoreBlock(nn.Module):
            """
            Core Convolutional block. 1 2d Conv followed by norm + ReLU
            """
            def __init__(self, in_channels:int, out_channels:int, kernel_size:int=3, stride:int=1):
                super().__init__()

                padding = (kernel_size - 1) // 2
                self.model = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU()
                )
            
            def forward(self, x):
                return self.model(x)

        # Core building blocks for backbone model
        class DownBlock(nn.Module):
            """
            Block for part of Encoder part of U-Net CNN. 
            """
            def __init__(self, in_channels, out_channels, n_convs_per_down_block=2, skip=True, kernel_size=3, stride=1):
                super().__init__()

                self.n_convs_per_down_block = n_convs_per_down_block
                self.in_channels = in_channels
                self.out_channels = out_channels
                self.skip = skip
                self.kernel_size = kernel_size
                self.stride = stride

                # First skip will always be changing channels - the rest just go from out_channels to out_channels
                self.skip_conns = nn.ModuleList()
                if (self.skip):
                    self.skip_conns.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0))
                    for _ in range(1, n_convs_per_down_block - 1):
                        self.skip_conns.append(nn.Identity())

                self.layers = nn.ModuleList()
                for _ in range(n_convs_per_down_block):
                    self.layers.append(CoreBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride))
                    
                    # Channels increase once then stay the same
                    in_channels = out_channels 
                
                # Add a 2x2 maxpool per paper
                self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
            

            def forward(self, x):
                """
                forward does not apply max pool
                """
                y = x
                skip_out = None
                for i in range(len(self.layers)):
                    if (i == 0 or not self.skip):
                        y = self.layers[i](y)
                    else:
                        y = self.layers[i](y + skip_out)

                    # Avoid going over length of array.
                    if (self.skip and i < len(self.layers) - 1):
                        skip_out = self.skip_conns[i](y)

                # do not max pool the final output (this will be done manually outside)
                return y

        class UnetEncode(nn.Module):
            def __init__(self, n_down_blocks, convs_per_down_block, image_input_channels, initial_output_channels, skip_within_blocks=True, skip_between_down_blocks=False, kernel_size=3, stride=1):
                super().__init__()

                self.skip_between_down_blocks = skip_between_down_blocks

                # When skipping between blocks, the number of channels changes every time
                self.skip_conns = nn.ModuleList()
                skip_in_channels = initial_output_channels
                skip_out_channels = initial_output_channels * 2
                if (self.skip_between_down_blocks):
                    for _ in range(1, n_down_blocks - 1):
                        self.skip_conns.append(nn.Conv2d(skip_in_channels, skip_out_channels, kernel_size=1, stride=stride, padding=0))
                        skip_in_channels = skip_out_channels
                        skip_out_channels *= 2
                    

                self.encode_layers = nn.ModuleList()
                encoder_block_in_channels = image_input_channels
                encoder_block_out_channels = initial_output_channels
                for _ in range(n_down_blocks):
                    self.encode_layers.append(DownBlock(n_convs_per_down_block=convs_per_down_block, 
                                                        in_channels=encoder_block_in_channels, out_channels=encoder_block_out_channels,
                                                        skip=skip_within_blocks, kernel_size=kernel_size, stride=stride
                                                        )
                                                        )
                    
                    encoder_block_in_channels = encoder_block_out_channels
                    encoder_block_out_channels *= 2

                self.layer_outputs = []


            def forward(self, x):
                # Reset the layer outputs
                self.layer_outputs = []
                skip_out = None
                y = x
                for i in range(len(self.encode_layers)):
                    if (i == 0 or not self.skip_between_down_blocks):
                        y = self.encode_layers[i](y)
                    else:
                        y = self.encode_layers[i](y + skip_out)

                    if (self.skip_between_down_blocks and i < len(self.encode_layers) - 1):
                        skip_out = self.skip_conns[i](y)
                    
                    # Before max pooling, store output of this down block for concatenations to Decoder
                    self.layer_outputs.append(y)
                    
                    # Apply max pooling to downsize H and W dims by factor of 2 before next encoder block
                    y = self.encode_layers[i].max_pool(y)

                return y

        class UpBlock(nn.Module):
            def __init__(self, total_input_channels, out_channels, n_convs_per_up_block=2, skip=True, kernel_size=3, stride=1, do_upconv=True):
                super().__init__()

                self.skip = skip
                self.n_convs_per_up_block = n_convs_per_up_block
                self.do_upconv = do_upconv

                # First skip will always be changing channels - the rest just go from out_channels to out_channels
                self.skip_conns = nn.ModuleList()
                if (self.skip):
                    self.skip_conns.append(nn.Conv2d(total_input_channels, out_channels, kernel_size=1, stride=stride, padding=0))
                    for _ in range(1, n_convs_per_up_block - 1):
                        self.skip_conns.append(nn.Identity())

                self.layers = nn.ModuleList()
                for _ in range(n_convs_per_up_block):
                    # print(f"\t within: {total_input_channels} --> {out_channels}")
                    self.layers.append(CoreBlock(total_input_channels, out_channels, kernel_size=kernel_size, stride=stride))
                    
                    # Channels decrease once then stay the same
                    total_input_channels = out_channels 
                
                # Instead of maxpooling, we need to upsample. We could do this with Upsample or a 2x2 upconv.
                # Paper does upconv, so I do upconv
                self.upconv = nn.ConvTranspose2d(total_input_channels, out_channels, kernel_size=2, stride=2) if self.do_upconv else None


            def forward(self, x):
                """
                forward does apply upconv
                """
                skip_out = None
                y = x
                for i in range(len(self.layers)):
                    if (i == 0 or not self.skip):
                        y = self.layers[i](y)
                    else:
                        y = self.layers[i](y + skip_out)

                    if (self.skip and i < len(self.layers) - 1):
                        skip_out = self.skip_conns[i](y)

                # Unlike in DownBlocks where we had to wait on applying max pool, we can safely return the upconvoluted output
                return self.upconv(y) if self.do_upconv else y

        class UnetDecode(nn.Module):
            def __init__(self, 
                         n_up_blocks:int, 
                         convs_per_up_block:int, 
                         input_channels:int, 
                         skip_within_blocks:bool=True, 
                         skip_between_up_blocks:bool=False, 
                         kernel_size:int=3, 
                         stride:int=1
                         ):
                super().__init__()
                
                self.skip_between_up_blocks = skip_between_up_blocks

                # When skipping between blocks, the number of channels changes every time
                self.skip_conns = nn.ModuleList()
                skip_in_channels = input_channels // 2
                skip_out_channels = skip_in_channels // 2
                if (self.skip_between_up_blocks):
                    for _ in range(1, n_up_blocks - 1):
                        self.skip_conns.append(nn.Conv2d(skip_in_channels, skip_out_channels, kernel_size=1, stride=stride, padding=0))
                        skip_in_channels = skip_out_channels
                        skip_out_channels = skip_out_channels // 2
                    

                self.decode_layers = nn.ModuleList()
                # This should be the number of channels after concatenation
                decoder_block_in_channels = input_channels #  512

                # This is because our encoder block will decrease channels by factor of 2, then we upsample via a transpose conv2d to decrease it by a power of 2 again
                decoder_block_out_channels = input_channels // 4 
                do_upconv = True 
                for i in range(n_up_blocks):
                    # See U-Net diagram. On final up-convolution block, we don't have an upconv at the end. This also means our output channels will be twice as large
                    if (i == n_up_blocks - 1):
                        do_upconv = False 
                        decoder_block_out_channels *= 2

                    self.decode_layers.append(UpBlock(
                        total_input_channels=decoder_block_in_channels, 
                        out_channels=decoder_block_out_channels, 
                        n_convs_per_up_block=convs_per_up_block, 
                        skip=skip_within_blocks, 
                        kernel_size=kernel_size, 
                        stride=stride,
                        do_upconv= do_upconv
                        )
                    )

                    # Output of this layer gets concatenated with skip connection
                    decoder_block_in_channels = decoder_block_out_channels * 2 # 256
                    decoder_block_out_channels = (decoder_block_in_channels // 4) 
            

            def forward(self, x, encoder_skip_conns):
                y = x 
                skip_out = None
                for i in range(len(self.decode_layers)):
                    # Concatenate along the channel dimensions (each input is B x C x W x H)
                    # Need to crop skip connection input to make sure it's the right height and width
                    y = torch.cat([y, encoder_skip_conns[-1 -i][:, :, :y.shape[2], :y.shape[3]]], dim=1)

                    if (i == 0 or not self.skip_between_up_blocks):
                        y = self.decode_layers[i](y)
                    else:
                        y = self.decode_layers[i](y + skip_out)
                    
                    if (self.skip_between_up_blocks and i < len(self.decode_layers) - 1):
                        skip_out = self.skip_conns[i](y)

                return y

        # Encoder half of U-Net
        self.encoder = UnetEncode(n_down_blocks=num_blocks, 
                                  convs_per_down_block=num_convs_per_block, 
                                  image_input_channels=in_channels, 
                                  initial_output_channels=initial_output_channels,
                                  skip_within_blocks=skip_within_blocks, 
                                  skip_between_down_blocks=skip_between_blocks
                                  )
        

        # Setup the bottleneck
        bottleneck_input_channels = initial_output_channels 
        for _ in range(1, num_blocks):
            # Each encoder block increases number of channels by factor of 2
            bottleneck_input_channels *= 2
        bottleneck_output_channels = bottleneck_input_channels * 2


        # Bottleneck is a 2-block CNN which then gets up sampled as input to Decoder.
        # Honestly this is the part I'm least understanding of, it's what differentiates autoencoder setups
        self.bottleneck = nn.Sequential(
            CoreBlock(bottleneck_input_channels, bottleneck_output_channels),
            CoreBlock(bottleneck_output_channels, bottleneck_output_channels),
            nn.ConvTranspose2d(bottleneck_output_channels, bottleneck_input_channels, kernel_size=2, stride=2)
        )

        # Conv transpose in bottlneck will get us back up to same C as input to bottleneck (output of encoder)
        # We multiply by 2 due to concatenation of this output with same C sized output of decoder.
        decoder_input_channels = bottleneck_input_channels * 2

        # Decoder half of U-Net
        self.decoder = UnetDecode(n_up_blocks=num_blocks,
                                  convs_per_up_block=num_convs_per_block,
                                  input_channels=decoder_input_channels,
                                  skip_within_blocks=skip_within_blocks,
                                  skip_between_up_blocks=skip_between_blocks
                                  )
        



        # Defining network head for segmentation - almost one to one for paper
        self.mask_head = nn.Sequential(
            CoreBlock(initial_output_channels, out_channels=initial_output_channels, kernel_size=3, stride=1),
            nn.Conv2d(initial_output_channels, self.num_classes, kernel_size=1, stride=1)
        )

        # Network head for depth regression. Same idea as paper's network head for segmentation
        self.depth_head = nn.Sequential(
            CoreBlock(initial_output_channels, out_channels=initial_output_channels, kernel_size=3, stride=1),
            nn.Conv2d(initial_output_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )



    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used in training, takes an image and returns raw logits and raw depth.
        This is what the loss functions use as input.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.FloatTensor, torch.FloatTensor):
                - logits (b, num_classes, h, w)
                - depth (b, h, w)
        """
        # optional: normalizes the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # --------------------------- U-Net Backbone ---------------------------
        # First pass through the decoder
        encoder_out = self.encoder(z)
        pre_pool_outputs = self.encoder.layer_outputs

        bottleneck_output = self.bottleneck(encoder_out)
        unet_backbone_output = self.decoder(bottleneck_output, pre_pool_outputs)
        # ----------------------------------------------------------------------
        
        # Output of unet backbone will be of shape B x initial_input_dim x H x W
        mask_output = self.mask_head(unet_backbone_output)
        depth_output = self.depth_head(unet_backbone_output).squeeze(1)

        return mask_output, depth_output



    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used for inference, takes an image and returns class labels and normalized depth.
        This is what the metrics use as input (this is what the grader will use!).

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.LongTensor, torch.FloatTensor):
                - pred: class labels {0, 1, 2} with shape (b, h, w)
                - depth: normalized depth [0, 1] with shape (b, h, w)
        """
        logits, raw_depth = self(x)
        pred = logits.argmax(dim=1)

        # Optional additional post-processing for depth only if needed
        depth = raw_depth

        return pred, depth


MODEL_FACTORY = {
    "classifier": Classifier,
    "detector": Detector,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) == m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def debug_model(batch_size: int = 1):
    """
    Test your model implementation

    Feel free to add additional checks to this function -
    this function is NOT used for grading
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_batch = torch.rand(batch_size, 3, 64, 64).to(device)

    print(f"Input shape: {sample_batch.shape}")

    model = load_model("classifier", in_channels=3, num_classes=6).to(device)
    output = model(sample_batch)

    # should output logits (b, num_classes)
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    debug_model()
