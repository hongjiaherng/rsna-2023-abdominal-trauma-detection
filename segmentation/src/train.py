import os
import lightning.pytorch as pl
import torch.utils.data

import rsna_datasets
import rsna_transforms
import models
from models.segresnet import SegResNet3D


def main(**configs):
    # Define transforms
    transforms = {
        "train": {
            "volume": rsna_transforms.volume_transforms(
                crop_strategy="oversample",
                voxel_spacing=configs.get("voxel_spacing"),
                volume_size=configs.get("volume_size"),
                axcodes=configs.get("axcodes"),
                streaming=configs.get("streaming"),
            ),
            "mask": rsna_transforms.mask_transforms(
                crop_strategy="oversample",
                voxel_spacing=configs.get("voxel_spacing"),
                volume_size=configs.get("volume_size"),
                axcodes=configs.get("axcodes"),
                streaming=configs.get("streaming"),
            ),
        },
        "test": {
            "volume": rsna_transforms.volume_transforms(
                crop_strategy="none",
                voxel_spacing=configs.get("voxel_spacing"),
                volume_size=configs.get("volume_size"),
                axcodes=configs.get("axcodes"),
                streaming=configs.get("streaming"),
            ),
            "mask": rsna_transforms.mask_transforms(
                crop_strategy="none",
                voxel_spacing=configs.get("voxel_spacing"),
                volume_size=configs.get("volume_size"),
                axcodes=configs.get("axcodes"),
                streaming=configs.get("streaming"),
            ),
        },
    }

    # Initialize datasets and dataloaders
    train_ds = rsna_datasets.Segmentation3DDataset(
        split="train",
        streaming=configs.get("streaming"),
        volume_transforms=transforms["train"]["volume"],
        mask_transforms=transforms["train"]["mask"],
        test_size=configs.get("test_size"),
        random_state=configs.get("random_state"),
    )
    test_ds = rsna_datasets.Segmentation3DDataset(
        split="test",
        streaming=configs.get("streaming"),
        volume_transforms=transforms["test"]["volume"],
        mask_transforms=transforms["test"]["mask"],
        test_size=configs.get("test_size"),
        random_state=configs.get("random_state"),
    )
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=configs.get("train_batch_size"),
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
    )
    test_dl = torch.utils.data.DataLoader(
        test_ds,
        batch_size=configs.get("test_batch_size"),
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
    )

    # Initialize model and trainer
    model = SegResNet3D(
        pretrained=configs.get("bootstrap_from_pretrained"),
        pretrained_path=configs.get("pretrained_path"),
    )
    trainer = pl.Trainer(
        max_epochs=configs.get("max_epochs"),
    )
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=test_dl)


if __name__ == "__main__":
    bootstrap_from_pretrained = True
    pretrained_path = os.path.abspath(
        os.path.join(__file__, "../../pretrained/segresnet/model_lowres.pt")
    )
    max_epochs = 2
    train_batch_size = 4
    test_batch_size = 1
    streaming = False
    test_size = 0.1
    random_state = 42

    volume_size = (96, 96, 96)
    voxel_spacing = (3.0, 3.0, 3.0)
    axcodes = "RAS"

    main(
        max_epochs=max_epochs,
        bootstrap_from_pretrained=bootstrap_from_pretrained,
        pretrained_path=pretrained_path,
        train_batch_size=train_batch_size,
        test_batch_size=test_batch_size,
        streaming=streaming,
        test_size=test_size,
        random_state=random_state,
        volume_size=volume_size,
        voxel_spacing=voxel_spacing,
        axcodes=axcodes,
    )
