import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import monai.losses
import monai.metrics
import monai.networks.nets
import monai.inferers
import torchmetrics.classification


class SegResNet3D(pl.LightningModule):
    def __init__(
        self,
        pretrained=False,
        pretrained_path=None,
        volume_size=(96, 96, 96),
        lr=0.0001,
        weight_decay=1e-5,
    ):
        super().__init__()

        self.lr = lr
        self.weight_decay = weight_decay

        self.criterion = monai.losses.DiceCELoss(
            to_onehot_y=True, softmax=True, include_background=True, reduction="mean"
        )  # Expect: y_pred is (B, C, H, W, D) logits, y is (B, 1, H, W, D) class indices
        self.dice_metric = monai.metrics.DiceMetric(
            include_background=True, reduction="mean"
        )  # Expect: y_pred is (B, C, H, W, D) one-hot class indices, y is (B, 1, H, W, D) class indices
        self.accuracy_metric = torchmetrics.classification.MulticlassAccuracy(
            num_classes=6, average="macro"
        )  # Expect: y_pred is (B, C, H, W, D) class probabilities, y is (B, H, W, D) class indices
        self.sw_inferer = monai.inferers.SlidingWindowInferer(
            roi_size=volume_size, sw_batch_size=4, overlap=0.25
        )
        self.model = monai.networks.nets.SegResNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=6,
            init_filters=32,
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            dropout_prob=0.2,
        )

        if pretrained:
            pretrained_state_dict = torch.load(
                pretrained_path,
                map_location=torch.device("cpu"),
            )
            if (
                "conv_final.2.conv.weight" not in pretrained_state_dict.keys()
                or "conv_final.2.conv.bias" not in pretrained_state_dict.keys()
            ):
                raise ValueError("Unexpected pretrained weights.")

            # Remove the last layer of the pretrained weights
            del pretrained_state_dict["conv_final.2.conv.weight"]
            del pretrained_state_dict["conv_final.2.conv.bias"]

            # Load the weights except the last layer
            self.model.load_state_dict(pretrained_state_dict, strict=False)

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch["img"], batch["seg"]  # x: (B, 1, H, W, D), y: (B, 1, H, W, D)
        y_logit = self(x)  # y_logit: (B, C, H, W, D)
        y_prob = F.softmax(y_logit, dim=1)  # y_prob: (B, C, H, W, D)
        y_indices = torch.permute(
            F.one_hot(torch.argmax(y_prob, dim=1), num_classes=6), (0, 4, 1, 2, 3)
        )  # y_indices: (B, C, H, W, D)

        loss = self.criterion(y_logit, y)
        accuracy_score = self.accuracy_metric(y_prob, y.squeeze(1))
        self.dice_metric(y_indices, y)  # Compute dice score for each class
        dice_score = (
            self.dice_metric.aggregate()
        )  # Compute mean dice score over all classes
        self.dice_metric.reset()  # Reset the metric calculated on the current batch

        metrics_step = {
            "train_loss": loss,
            "train_acc": accuracy_score,
            "train_dice": dice_score,
        }

        self.log_dict(metrics_step, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["img"], batch["seg"]  # x: (B, 1, H, W, D), y: (B, 1, H, W, D)
        # do inference in sliding window fashion because of GPU memory limitation
        # input for validation is not pre-cropped to designated ROI size, thus the input shape would be a few times larger
        y_logit = self.sw_inferer(x, self)
        y_prob = F.softmax(y_logit, dim=1)
        y_indices = torch.permute(
            F.one_hot(torch.argmax(y_prob, dim=1), num_classes=6), (0, 4, 1, 2, 3)
        )

        loss = self.criterion(y_logit, y)
        accuracy_score = self.accuracy_metric(y_prob, y.squeeze(1))
        self.dice_metric(y_indices, y)  # Compute dice score for each class
        dice_score = (
            self.dice_metric.aggregate()
        )  # Compute mean dice score over all classes
        self.dice_metric.reset()  # Reset the metric calculated on the current batch

        metrics_step = {
            "val_loss": loss,
            "val_acc": accuracy_score,
            "val_dice": dice_score,
        }

        self.log_dict(metrics_step, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
