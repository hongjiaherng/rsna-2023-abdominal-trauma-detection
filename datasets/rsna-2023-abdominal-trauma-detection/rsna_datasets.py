from typing import Literal, TypedDict

import datasets
import torch
import monai.transforms
import torchvision
from torch.utils.data import IterableDataset

import rsna_transforms


class Segmentation3DDataset(IterableDataset):
    def __init__(
        self,
        split: Literal["train", "test"],
        streaming: bool = True,
        volume_transforms: monai.transforms.Compose = None,
        mask_transforms: monai.transforms.Compose = None,
        transform_configs: TypedDict(
            "",
            {
                "crop_strategy": Literal["oversample", "center", "random", "none"],
                "voxel_spacing": tuple[float, float, float],
                "volume_size": tuple[int, int, int],
                "axcodes": str,
            },
        ) = {
            "crop_strategy": "oversample",
            "voxel_spacing": (3.0, 3.0, 3.0),
            "volume_size": (96, 96, 96),
            "axcodes": "RAS",
        },
        test_size: float = 0.1,
        random_state: int = 42,
    ):
        self.hf_dataset = datasets.load_dataset(
            "jherng/rsna-2023-abdominal-trauma-detection",
            "segmentation",
            split=split,
            streaming=streaming,
            num_proc=4
            if not streaming
            else None,  # Use multiprocessing if not streaming to download faster
            test_size=test_size,
            random_state=random_state,
        )

        self.volume_transforms = volume_transforms or rsna_transforms.volume_transforms(
            crop_strategy=transform_configs["crop_strategy"],
            voxel_spacing=transform_configs["voxel_spacing"],
            volume_size=transform_configs["volume_size"],
            axcodes=transform_configs["axcodes"],
            streaming=streaming,
        )

        self.mask_transforms = mask_transforms or rsna_transforms.mask_transforms(
            crop_strategy=transform_configs["crop_strategy"],
            voxel_spacing=transform_configs["voxel_spacing"],
            volume_size=transform_configs["volume_size"],
            axcodes=transform_configs["axcodes"],
            streaming=streaming,
        )
        self.yield_extra_info = True  # For debugging purposes

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else -1

        if isinstance(self.hf_dataset, datasets.Dataset):
            start_idx = worker_id if worker_id != -1 else 0
            step_size = worker_info.num_workers if worker_id != -1 else 1

            for i in range(start_idx, len(self.hf_dataset), step_size):
                data = self.hf_dataset[i]
                yield from self._process_one_sample(data, worker_id=worker_id)
        else:
            for i, data in enumerate(self.hf_dataset):
                yield from self._process_one_sample(data, worker_id=worker_id)

    def _process_one_sample(self, data, worker_id):
        img_data = self.volume_transforms(
            {"img": data["img_path"], "metadata": data["metadata"]}
        )
        seg_data = self.mask_transforms({"seg": data["seg_path"]})

        img_data = [img_data] if not isinstance(img_data, (list, tuple)) else img_data
        seg_data = [seg_data] if not isinstance(seg_data, (list, tuple)) else seg_data

        for img, seg in zip(img_data, seg_data):
            to_yield = {
                "img": img["img"],
                "seg": seg["seg"],
            }
            if self.yield_extra_info:
                to_yield["worker_id"] = worker_id
                to_yield["series_id"] = data["metadata"]["series_id"]

            yield to_yield


class Classification3DDataset(IterableDataset):
    def __init__(
        self,
        split: Literal["train", "test"],
        streaming: bool = True,
        volume_transforms: monai.transforms.Compose = None,
        transform_configs: TypedDict(
            "",
            {
                "crop_strategy": Literal["oversample", "center", "random", "none"],
                "voxel_spacing": tuple[float, float, float],
                "volume_size": tuple[int, int, int],
                "axcodes": str,
            },
        ) = {
            "crop_strategy": "oversample",
            "voxel_spacing": (3.0, 3.0, 3.0),
            "volume_size": (96, 96, 96),
            "axcodes": "RAS",
        },
        test_size: float = 0.1,
        random_state: int = 42,
    ):
        self.hf_dataset = datasets.load_dataset(
            "jherng/rsna-2023-abdominal-trauma-detection",
            "classification",
            split=split,
            streaming=streaming,
            num_proc=4
            if not streaming
            else None,  # Use multiprocessing if not streaming to download faster
            test_size=test_size,
            random_state=random_state,
        )

        self.volume_transforms = volume_transforms or rsna_transforms.volume_transforms(
            crop_strategy=transform_configs["crop_strategy"],
            voxel_spacing=transform_configs["voxel_spacing"],
            volume_size=transform_configs["volume_size"],
            axcodes=transform_configs["axcodes"],
            streaming=streaming,
        )

        self.yield_extra_info = True

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else -1

        if isinstance(self.hf_dataset, datasets.Dataset):
            start_idx = worker_id if worker_id != -1 else 0
            step_size = worker_info.num_workers if worker_id != -1 else 1

            for i in range(start_idx, len(self.hf_dataset), step_size):
                data = self.hf_dataset[i]
                yield from self._process_one_sample(data, worker_id=worker_id)
        else:
            for i, data in enumerate(self.hf_dataset):
                yield from self._process_one_sample(data, worker_id=worker_id)

    def _process_one_sample(self, data, worker_id):
        img_data = self.volume_transforms(
            {"img": data["img_path"], "metadata": data["metadata"]}
        )
        img_data = [img_data] if not isinstance(img_data, (list, tuple)) else img_data

        for img in img_data:
            to_yield = {
                "img": img["img"],
                "bowel": data["bowel"],
                "extravasation": data["extravasation"],
                "kidney": data["kidney"],
                "liver": data["liver"],
                "spleen": data["spleen"],
                "any_injury": data["any_injury"],
            }

            if self.yield_extra_info:
                to_yield["worker_id"] = worker_id
                to_yield["series_id"] = data["metadata"]["series_id"]

            yield to_yield


class MaskedClassification3DDataset(IterableDataset):
    def __init__(
        self,
        split: Literal["train", "test"],
        streaming: bool = True,
        volume_transforms: monai.transforms.Compose = None,
        mask_transforms: monai.transforms.Compose = None,
        transform_configs: TypedDict(
            "",
            {
                "crop_strategy": Literal["oversample", "center", "random", "none"],
                "voxel_spacing": tuple[float, float, float],
                "volume_size": tuple[int, int, int],
                "axcodes": str,
            },
        ) = {
            "crop_strategy": "oversample",
            "voxel_spacing": (3.0, 3.0, 3.0),
            "volume_size": (96, 96, 96),
            "axcodes": "RAS",
        },
        test_size: float = 0.1,
        random_state: int = 42,
    ):
        self.hf_dataset = datasets.load_dataset(
            "jherng/rsna-2023-abdominal-trauma-detection",
            "classification-with-mask",
            split=split,
            streaming=streaming,
            num_proc=4
            if not streaming
            else None,  # Use multiprocessing if not streaming to download faster
            test_size=test_size,
            random_state=random_state,
        )

        self.volume_transforms = volume_transforms or rsna_transforms.volume_transforms(
            crop_strategy=transform_configs["crop_strategy"],
            voxel_spacing=transform_configs["voxel_spacing"],
            volume_size=transform_configs["volume_size"],
            axcodes=transform_configs["axcodes"],
            streaming=streaming,
        )
        self.mask_transforms = mask_transforms or rsna_transforms.mask_transforms(
            crop_strategy=transform_configs["crop_strategy"],
            voxel_spacing=transform_configs["voxel_spacing"],
            volume_size=transform_configs["volume_size"],
            axcodes=transform_configs["axcodes"],
            streaming=streaming,
        )

        self.yield_extra_info = True

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else -1

        if isinstance(self.hf_dataset, datasets.Dataset):
            start_idx = worker_id if worker_id != -1 else 0
            step_size = worker_info.num_workers if worker_id != -1 else 1

            for i in range(start_idx, len(self.hf_dataset), step_size):
                data = self.hf_dataset[i]
                yield from self._process_one_sample(data, worker_id=worker_id)
        else:
            for i, data in enumerate(self.hf_dataset):
                yield from self._process_one_sample(data, worker_id=worker_id)

    def _process_one_sample(self, data, worker_id):
        img_data = self.volume_transforms(
            {"img": data["img_path"], "metadata": data["metadata"]}
        )
        seg_data = self.mask_transforms({"seg": data["seg_path"]})
        img_data = [img_data] if not isinstance(img_data, (list, tuple)) else img_data
        seg_data = [seg_data] if not isinstance(seg_data, (list, tuple)) else seg_data

        for img, seg in zip(img_data, seg_data):
            to_yield = {
                "img": img["img"],
                "seg": seg["seg"],
                "bowel": data["bowel"],
                "extravasation": data["extravasation"],
                "kidney": data["kidney"],
                "liver": data["liver"],
                "spleen": data["spleen"],
                "any_injury": data["any_injury"],
            }

            if self.yield_extra_info:
                to_yield["worker_id"] = worker_id
                to_yield["series_id"] = data["metadata"]["series_id"]

            yield to_yield


class Segmentation2DDataset(IterableDataset):
    def __init__(
        self,
        split: Literal["train", "test"],
        streaming: bool = True,
        volume_transforms: monai.transforms.Compose = None,
        mask_transforms: monai.transforms.Compose = None,
        slice_transforms: torchvision.transforms.Compose = None,
        volume_transform_configs: TypedDict(
            "",
            {
                "crop_strategy": Literal["oversample", "center", "random", "none"],
                "voxel_spacing": tuple[float, float, float],
                "volume_size": tuple[int, int, int],
                "axcodes": str,
            },
        ) = {
            "crop_strategy": "none",
            "voxel_spacing": (3.0, 3.0, 3.0),
            "volume_size": None,
            "axcodes": "RAS",
        },
        slice_transform_configs: TypedDict(
            "",
            {
                "crop_strategy": Literal["ten", "five", "center", "random"],
                "shorter_edge_length": int,
                "slice_size": tuple[int, int],
            },
        ) = {
            "crop_strategy": "center",
            "shorter_edge_length": 256,
            "slice_size": (224, 224),
        },
        test_size: float = 0.1,
        random_state: int = 42,
    ):
        self.hf_dataset = datasets.load_dataset(
            "jherng/rsna-2023-abdominal-trauma-detection",
            "segmentation",
            split=split,
            streaming=streaming,
            num_proc=4
            if not streaming
            else None,  # Use multiprocessing if not streaming to download faster
            test_size=test_size,
            random_state=random_state,
        )

        self.volume_transforms = volume_transforms or rsna_transforms.volume_transforms(
            crop_strategy=volume_transform_configs["crop_strategy"],
            voxel_spacing=volume_transform_configs["voxel_spacing"],
            volume_size=volume_transform_configs["volume_size"],
            axcodes=volume_transform_configs["axcodes"],
            streaming=streaming,
        )
        self.mask_transforms = mask_transforms or rsna_transforms.mask_transforms(
            crop_strategy=volume_transform_configs["crop_strategy"],
            voxel_spacing=volume_transform_configs["voxel_spacing"],
            volume_size=volume_transform_configs["volume_size"],
            axcodes=volume_transform_configs["axcodes"],
            streaming=streaming,
        )
        self.slice_transforms = slice_transforms or rsna_transforms.slice_transforms(
            crop_strategy=slice_transform_configs["crop_strategy"],
            shorter_edge_length=slice_transform_configs["shorter_edge_length"],
            slice_size=slice_transform_configs["slice_size"],
        )
        self.yield_extra_info = True  # For debugging purposes

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else -1

        if isinstance(self.hf_dataset, datasets.Dataset):
            start_idx = worker_id if worker_id != -1 else 0
            step_size = worker_info.num_workers if worker_id != -1 else 1

            for i in range(start_idx, len(self.hf_dataset), step_size):
                data = self.hf_dataset[i]
                yield from self._process_one_sample(data, worker_id=worker_id)
        else:
            for i, data in enumerate(self.hf_dataset):
                yield from self._process_one_sample(data, worker_id=worker_id)

    def _process_one_sample(self, data, worker_id):
        vol_img_data = self.volume_transforms(
            {"img": data["img_path"], "metadata": data["metadata"]}
        )
        vol_seg_data = self.mask_transforms({"seg": data["seg_path"]})
        vol_img_data = (
            [vol_img_data]
            if not isinstance(vol_img_data, (list, tuple))
            else vol_img_data
        )
        vol_seg_data = (
            [vol_seg_data]
            if not isinstance(vol_seg_data, (list, tuple))
            else vol_seg_data
        )

        for vol_img, vol_seg in zip(vol_img_data, vol_seg_data):
            slice_len = vol_img["img"].size()[-1]
            for i in range(slice_len):
                slice_img_data = self.slice_transforms(vol_img["img"][..., i])
                slice_seg_data = self.slice_transforms(vol_seg["seg"][..., i])

                slice_img_data = (
                    [slice_img_data]
                    if not isinstance(slice_img_data, (list, tuple))
                    else slice_img_data
                )
                slice_seg_data = (
                    [slice_seg_data]
                    if not isinstance(slice_seg_data, (list, tuple))
                    else slice_seg_data
                )

                for slice_img, slice_seg in zip(slice_img_data, slice_seg_data):
                    to_yield = {
                        "img": slice_img,
                        "seg": slice_seg,
                    }
                    if self.yield_extra_info:
                        to_yield["worker_id"] = worker_id
                        to_yield["series_id"] = data["metadata"]["series_id"]

                    yield to_yield


class Classification2DDataset(IterableDataset):
    def __init__(
        self,
        split: Literal["train", "test"],
        streaming: bool = True,
        volume_transforms: monai.transforms.Compose = None,
        slice_transforms: torchvision.transforms.Compose = None,
        volume_transform_configs: TypedDict(
            "",
            {
                "crop_strategy": Literal["oversample", "center", "random", "none"],
                "voxel_spacing": tuple[float, float, float],
                "volume_size": tuple[int, int, int],
                "axcodes": str,
            },
        ) = {
            "crop_strategy": "none",
            "voxel_spacing": (3.0, 3.0, 3.0),
            "volume_size": None,
            "axcodes": "RAS",
        },
        slice_transform_configs: TypedDict(
            "",
            {
                "crop_strategy": Literal["ten", "five", "center", "random"],
                "shorter_edge_length": int,
                "slice_size": tuple[int, int],
            },
        ) = {
            "crop_strategy": "center",
            "shorter_edge_length": 256,
            "slice_size": (224, 224),
        },
        test_size: float = 0.1,
        random_state: int = 42,
    ):
        self.hf_dataset = datasets.load_dataset(
            "jherng/rsna-2023-abdominal-trauma-detection",
            "classification",
            split=split,
            streaming=streaming,
            num_proc=4
            if not streaming
            else None,  # Use multiprocessing if not streaming to download faster
            test_size=test_size,
            random_state=random_state,
        )

        self.volume_transforms = volume_transforms or rsna_transforms.volume_transforms(
            crop_strategy=volume_transform_configs["crop_strategy"],
            voxel_spacing=volume_transform_configs["voxel_spacing"],
            volume_size=volume_transform_configs["volume_size"],
            axcodes=volume_transform_configs["axcodes"],
            streaming=streaming,
        )
        self.slice_transforms = slice_transforms or rsna_transforms.slice_transforms(
            crop_strategy=slice_transform_configs["crop_strategy"],
            shorter_edge_length=slice_transform_configs["shorter_edge_length"],
            slice_size=slice_transform_configs["slice_size"],
        )
        self.yield_extra_info = True  # For debugging purposes

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else -1

        if isinstance(self.hf_dataset, datasets.Dataset):
            start_idx = worker_id if worker_id != -1 else 0
            step_size = worker_info.num_workers if worker_id != -1 else 1

            for i in range(start_idx, len(self.hf_dataset), step_size):
                data = self.hf_dataset[i]
                yield from self._process_one_sample(data, worker_id=worker_id)
        else:
            for i, data in enumerate(self.hf_dataset):
                yield from self._process_one_sample(data, worker_id=worker_id)

    def _process_one_sample(self, data, worker_id):
        vol_img_data = self.volume_transforms(
            {"img": data["img_path"], "metadata": data["metadata"]}
        )
        vol_img_data = (
            [vol_img_data]
            if not isinstance(vol_img_data, (list, tuple))
            else vol_img_data
        )

        for vol_img in vol_img_data:
            slice_len = vol_img["img"].size()[-1]
            for i in range(slice_len):
                slice_img_data = self.slice_transforms(vol_img["img"][..., i])

                slice_img_data = (
                    [slice_img_data]
                    if not isinstance(slice_img_data, (list, tuple))
                    else slice_img_data
                )

                for slice_img in slice_img_data:
                    to_yield = {
                        "img": slice_img,
                        "bowel": data["bowel"],
                        "extravasation": data["extravasation"],
                        "kidney": data["kidney"],
                        "liver": data["liver"],
                        "spleen": data["spleen"],
                        "any_injury": data["any_injury"],
                    }
                    if self.yield_extra_info:
                        to_yield["worker_id"] = worker_id
                        to_yield["series_id"] = data["metadata"]["series_id"]

                    yield to_yield


class MaskedClassification2DDataset(IterableDataset):
    def __init__(
        self,
        split: Literal["train", "test"],
        streaming: bool = True,
        volume_transforms: monai.transforms.Compose = None,
        mask_transforms: monai.transforms.Compose = None,
        slice_transforms: torchvision.transforms.Compose = None,
        volume_transform_configs: TypedDict(
            "",
            {
                "crop_strategy": Literal["oversample", "center", "random", "none"],
                "voxel_spacing": tuple[float, float, float],
                "volume_size": tuple[int, int, int],
                "axcodes": str,
            },
        ) = {
            "crop_strategy": "none",
            "voxel_spacing": (3.0, 3.0, 3.0),
            "volume_size": None,
            "axcodes": "RAS",
        },
        slice_transform_configs: TypedDict(
            "",
            {
                "crop_strategy": Literal["ten", "five", "center", "random"],
                "shorter_edge_length": int,
                "slice_size": tuple[int, int],
            },
        ) = {
            "crop_strategy": "center",
            "shorter_edge_length": 256,
            "slice_size": (224, 224),
        },
        test_size: float = 0.1,
        random_state: int = 42,
    ):
        self.hf_dataset = datasets.load_dataset(
            "jherng/rsna-2023-abdominal-trauma-detection",
            "classification-with-mask",
            split=split,
            streaming=streaming,
            num_proc=4
            if not streaming
            else None,  # Use multiprocessing if not streaming to download faster
            test_size=test_size,
            random_state=random_state,
        )

        self.volume_transforms = volume_transforms or rsna_transforms.volume_transforms(
            crop_strategy=volume_transform_configs["crop_strategy"],
            voxel_spacing=volume_transform_configs["voxel_spacing"],
            volume_size=volume_transform_configs["volume_size"],
            axcodes=volume_transform_configs["axcodes"],
            streaming=streaming,
        )

        self.mask_transforms = mask_transforms or rsna_transforms.mask_transforms(
            crop_strategy=volume_transform_configs["crop_strategy"],
            voxel_spacing=volume_transform_configs["voxel_spacing"],
            volume_size=volume_transform_configs["volume_size"],
            axcodes=volume_transform_configs["axcodes"],
            streaming=streaming,
        )

        self.slice_transforms = slice_transforms or rsna_transforms.slice_transforms(
            crop_strategy=slice_transform_configs["crop_strategy"],
            shorter_edge_length=slice_transform_configs["shorter_edge_length"],
            slice_size=slice_transform_configs["slice_size"],
        )
        self.yield_extra_info = True  # For debugging purposes

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else -1

        if isinstance(self.hf_dataset, datasets.Dataset):
            start_idx = worker_id if worker_id != -1 else 0
            step_size = worker_info.num_workers if worker_id != -1 else 1

            for i in range(start_idx, len(self.hf_dataset), step_size):
                data = self.hf_dataset[i]
                yield from self._process_one_sample(data, worker_id=worker_id)
        else:
            for i, data in enumerate(self.hf_dataset):
                yield from self._process_one_sample(data, worker_id=worker_id)

    def _process_one_sample(self, data, worker_id):
        vol_img_data = self.volume_transforms(
            {"img": data["img_path"], "metadata": data["metadata"]}
        )
        vol_seg_data = self.mask_transforms({"seg": data["seg_path"]})
        vol_img_data = (
            [vol_img_data]
            if not isinstance(vol_img_data, (list, tuple))
            else vol_img_data
        )
        vol_seg_data = (
            [vol_seg_data]
            if not isinstance(vol_seg_data, (list, tuple))
            else vol_seg_data
        )

        for vol_img, vol_seg in zip(vol_img_data, vol_seg_data):
            slice_len = vol_img["img"].size()[-1]
            for i in range(slice_len):
                slice_img_data = self.slice_transforms(vol_img["img"][..., i])
                slice_seg_data = self.slice_transforms(vol_seg["seg"][..., i])

                slice_img_data = (
                    [slice_img_data]
                    if not isinstance(slice_img_data, (list, tuple))
                    else slice_img_data
                )
                slice_seg_data = (
                    [slice_seg_data]
                    if not isinstance(slice_seg_data, (list, tuple))
                    else slice_seg_data
                )

                for slice_img, slice_seg in zip(slice_img_data, slice_seg_data):
                    to_yield = {
                        "img": slice_img,
                        "seg": slice_seg,
                        "bowel": data["bowel"],
                        "extravasation": data["extravasation"],
                        "kidney": data["kidney"],
                        "liver": data["liver"],
                        "spleen": data["spleen"],
                        "any_injury": data["any_injury"],
                    }
                    if self.yield_extra_info:
                        to_yield["worker_id"] = worker_id
                        to_yield["series_id"] = data["metadata"]["series_id"]

                    yield to_yield

