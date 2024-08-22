import torch.utils.tensorboard as tb
import torch 
import numpy as np

import os
import argparse
from pathlib import Path
from datetime import datetime

from unet import UnetModel


def train(lr:float=1e-4, 
          weight_decay:float=0.01, 
          num_epochs:int=50, 
          batch_size:int=64, 
          augment_dataset:bool=False, 
          shuffle:bool=True, 
          num_workers:int=4, 
          seed:int=42,
          **kwargs):
    
    if torch.cuda.is_available():
        print("CUDA found, using GPU")
        device = torch.device("cuda")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)


    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path("logs") / f"{datetime.now().strftime('%m%d_%H%M%S')}"
    checkpoint_dir = Path("checkpoints") / f"{datetime.now().strftime('%m%d_%H%M%S')}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Tensorboard
    writer = tb.SummaryWriter(log_dir)
    
    # TODO: Define your dataset/dataloader here
    train_data_loader = ...
    val_data_loader = ...
    

    # Defaults are 3 channels, 3 classes, 2 blocks -- change for grader if changed here.
    model = UnetModel(in_channels=3,
                     num_classes=3, 
                     initial_output_channels=64,
                     num_blocks=2,
                     num_convs_per_block=2,
                     skip_between_blocks=False,
                     skip_within_blocks=False
                     ).to(device)
    
    # You can add a computation graph for your model here if you want. Change your torch.zeros to shape of your input
    # writer.add_graph(model, torch.zeros(1, 3, 96, 128))
    # writer.flush()

    writer.add_hparams({
        "lr" : lr,
        "weight_decay" : weight_decay,
        "num_epochs" : num_epochs,
        "batch_size" : batch_size,
        "num_blocks" : model.num_blocks,
        "augment_dataset" : augment_dataset,
        "shuffle" : True,
        "seed" : seed
    }, {})
    writer.flush()

    # Define loss functions here
    mask_loss_fn = torch.nn.functional.cross_entropy
    regression_loss_fn = torch.nn.functional.l1_loss

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # For tensorboard
    global_step = 0
    
    for epoch in range(num_epochs):
        model.train()
        metrics = {"train_mask_acc": [], "train_mask_iou": [], "val_mask_acc": [], "val_mask_iou": [], "train_depth_mae": [], "val_depth_mae": []}
        for train_instance in train_data_loader:
            X_train = train_instance["image"].to(device)
            y_depth = train_instance["depth"].to(device)
            y_mask = train_instance["track"].to(device)
            
            optimizer.zero_grad()
            y_pred_mask, y_pred_depth = model(X_train)

            # NOTE: you can adjust class weights here. This is helpful for tasks where background pixels are common
            mask_loss = mask_loss_fn(y_pred_mask, y_mask)
            depth_loss = regression_loss_fn(y_pred_depth, y_depth)
            total_loss = mask_loss + depth_loss
            
            total_loss.backward()
            optimizer.step()

            metrics["train_depth_mae"].append(depth_loss.item())
            global_step += 1


        # After each epoch store this
        # TODO: Calculate IoU here + accuracy here if you want to log it for tensorboard
        # writer.add_scalar("train/mask_acc", train_acc, global_step)
        # writer.add_scalar("train/mask_iou", train_iou, global_step)
        # writer.flush()

        model.eval()
        with torch.inference_mode():
            for val_instance in val_data_loader:
                X_val = val_instance["image"].to(device)
                y_mask = val_instance["track"].to(device)
                y_depth = val_instance["depth"].to(device)

                X_val = X_val.to(device)
                y_mask = y_mask.to(device)
                y_depth = y_depth.to(device)
                y_pred_mask, y_pred_depth = model(X_val)

                mask_loss = mask_loss_fn(y_pred_mask, y_mask)
                depth_loss = regression_loss_fn(y_pred_depth, y_depth)
                total_loss = mask_loss + depth_loss

                metrics["val_depth_mae"].append(depth_loss.item())

        # log average train and val accuracy to tensorboard
        epoch_train_depth_mae = torch.as_tensor(metrics["train_depth_mae"]).mean()
        epoch_val_depth_mae = torch.as_tensor(metrics["val_depth_mae"]).mean()

        # TODO: Calculate IoU here + accuracy here if you want to log it for tensorboard
        # writer.add_scalar("val/mask_acc", val_acc, global_step)
        # writer.add_scalar("val/mask_iou", val_iou, global_step)
        
        writer.add_scalar("train/depth_mae", epoch_train_depth_mae, global_step)
        writer.add_scalar("val/depth_mae", epoch_val_depth_mae, global_step)
        writer.flush()

        if epoch == 0 or (epoch) % 10 == 0:
            # Save model checkpoint
            torch.save(model.state_dict(), checkpoint_dir / f"model_epoch_{epoch}.th")

            # Uncomment the commented values here if you change the TODOs above.
            print(
                f"Epoch {epoch + 1:2d} / {num_epochs:2d}: "
                # f"train_iou={train_iou:.4f} "
                # f"train_acc={train_acc:.4f} "
                f"train_mae={epoch_train_depth_mae:.4f}"
                # f"\tval_iou={val_iou:.4f} "
                # f"val_acc={val_acc:.4f} "
                f"val_mae={epoch_val_depth_mae:.4f} "
                f"\tsaved to /model_epoch_{epoch}.th"
            )

            
    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), checkpoint_dir / f"classifier_epoch_final.th")
    print(f"Copy of final model weights saved to {checkpoint_dir}")

            


if (__name__ == "__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--augment_dataset", type=bool, default=False)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    

    # pass all arguments to train
    train(**vars(parser.parse_args()))

    

    