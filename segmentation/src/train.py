import os
import sys
import lightning.pytorch as pl
import torch.utils.data
from lightning.pytorch.loggers import WandbLogger

from models.segresnet import SegResNet3D


def import_custom_modules_to_globals():
    global rsna_datasets
    global rsna_transforms

    # Add datasets/rsna-2023-abdominal-trauma-detection to pythonpath
    sys.path.append(
        os.path.abspath(
            os.path.join(
                __file__, "../../../datasets/rsna-2023-abdominal-trauma-detection"
            )
        )
    )

    import rsna_datasets
    import rsna_transforms


def main(model_configs, train_configs, dataset_configs, preprocessing_configs):
    # Define transforms
    transforms = {
        "train": {
            "volume": rsna_transforms.volume_transforms(
                crop_strategy=preprocessing_configs.get("train_crop_strategy"),
                voxel_spacing=preprocessing_configs.get("voxel_spacing"),
                volume_size=preprocessing_configs.get("volume_size"),
                axcodes=preprocessing_configs.get("axcodes"),
                streaming=dataset_configs.get("streaming"),
            ),
            "mask": rsna_transforms.mask_transforms(
                crop_strategy=preprocessing_configs.get("train_crop_strategy"),
                voxel_spacing=preprocessing_configs.get("voxel_spacing"),
                volume_size=preprocessing_configs.get("volume_size"),
                axcodes=preprocessing_configs.get("axcodes"),
                streaming=dataset_configs.get("streaming"),
            ),
        },
        "test": {
            "volume": rsna_transforms.volume_transforms(
                crop_strategy=preprocessing_configs.get("test_crop_strategy"),
                voxel_spacing=preprocessing_configs.get("voxel_spacing"),
                volume_size=preprocessing_configs.get("volume_size"),
                axcodes=preprocessing_configs.get("axcodes"),
                streaming=dataset_configs.get("streaming"),
            ),
            "mask": rsna_transforms.mask_transforms(
                crop_strategy=preprocessing_configs.get("test_crop_strategy"),
                voxel_spacing=preprocessing_configs.get("voxel_spacing"),
                volume_size=preprocessing_configs.get("volume_size"),
                axcodes=preprocessing_configs.get("axcodes"),
                streaming=dataset_configs.get("streaming"),
            ),
        },
    }

    # Initialize datasets and dataloaders
    train_ds = rsna_datasets.Segmentation3DDataset(
        split="train",
        streaming=dataset_configs.get("streaming"),
        volume_transforms=transforms["train"]["volume"],
        mask_transforms=transforms["train"]["mask"],
        test_size=dataset_configs.get("test_size"),
        random_state=dataset_configs.get("random_state"),
    )
    test_ds = rsna_datasets.Segmentation3DDataset(
        split="test",
        streaming=dataset_configs.get("streaming"),
        volume_transforms=transforms["test"]["volume"],
        mask_transforms=transforms["test"]["mask"],
        test_size=dataset_configs.get("test_size"),
        random_state=dataset_configs.get("random_state"),
    )
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=train_configs.get("train_batch_size"),
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
    )
    test_dl = torch.utils.data.DataLoader(
        test_ds,
        batch_size=train_configs.get("test_batch_size"),
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
    )

    # Initialize model and trainer
    model = SegResNet3D(
        pretrained=model_configs.get("bootstrap_from_pretrained"),
        pretrained_path=model_configs.get("pretrained_path"),
        volume_size=preprocessing_configs.get("volume_size"),
        lr=train_configs.get("learning_rate"),
        weight_decay=train_configs.get("weight_decay"),
    )
    logger = WandbLogger(
        project="rsna-2023-abdominal-trauma-detection",
        save_dir=os.path.abspath(os.path.join(__file__, "../../")),
    )
    logger.experiment.config.update(
        {
            "model_configs": model_configs,
            "train_configs": train_configs,
            "dataset_configs": dataset_configs,
            "preprocessing_configs": preprocessing_configs,
        }
    )
    trainer = pl.Trainer(
        max_epochs=train_configs.get("max_epochs"),
        logger=logger,
        overfit_batches=5,
        log_every_n_steps=1,
        val_check_interval=1.0,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=2,
    )
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=test_dl)


if __name__ == "__main__":
    import_custom_modules_to_globals()

    # Model configs
    model_configs = {
        "bootstrap_from_pretrained": True,
        "pretrained_path": os.path.abspath(
            os.path.join(__file__, "../../pretrained/segresnet/model_lowres.pt")
        ),
    }

    # Training configs
    train_configs = {
        "max_epochs": 1000,
        "train_batch_size": 8,
        "test_batch_size": 1,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
    }

    # Dataset configs
    dataset_configs = {
        "streaming": False,
        "test_size": 0.1,
        "random_state": 42,
    }

    # Preprocessing configs
    preprocessing_configs = {
        "train_crop_strategy": "center",  # 'oversample', 'center', 'random', 'none'
        "test_crop_strategy": "none",
        "volume_size": (96, 96, 96),
        "voxel_spacing": (3.0, 3.0, 3.0),
        "axcodes": "RAS",
    }

    main(model_configs, train_configs, dataset_configs, preprocessing_configs)
