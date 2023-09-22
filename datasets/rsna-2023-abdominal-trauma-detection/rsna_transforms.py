from typing import Optional, Literal

from io import BytesIO
import numpy as np
import nibabel as nib
import torch
import torchvision
import monai
import monai.transforms
from indexed_gzip import IndexedGzipFile
from monai.data.image_reader import NibabelReader
from monai.transforms.io.array import switch_endianness
from monai.transforms.transform import MapTransform, Transform
from monai.data import MetaTensor
from monai.data.utils import correct_nifti_header_if_necessary
from monai.config import KeysCollection, DtypeLike
from monai.utils import (
    ImageMetaKey,
    convert_to_dst_type,
    ensure_tuple_rep,
    ensure_tuple,
)
from monai.utils.enums import PostFix
from huggingface_hub import HfFileSystem


class LoadNIfTIFromLocalCache(Transform):
    def __init__(
        self,
        dtype: DtypeLike | None = np.float32,
        ensure_channel_first: bool = False,
        simple_keys: bool = False,
        prune_meta_pattern: str | None = None,
        prune_meta_sep: str = ".",
    ):
        self.dtype = dtype
        self.ensure_channel_first = ensure_channel_first
        self.simple_keys = simple_keys
        self.pattern = prune_meta_pattern
        self.sep = prune_meta_sep

        self.reader = NibabelReader()

    def __call__(self, path: str):
        with open(path, mode="rb") as f:
            img = nib.Nifti1Image.from_stream(
                IndexedGzipFile(fileobj=BytesIO(f.read()))
            )

        img = correct_nifti_header_if_necessary(img)
        img_array, meta_data = self.reader.get_data(img)
        img_array = convert_to_dst_type(img_array, dst=img_array, dtype=self.dtype)[0]
        if not isinstance(meta_data, dict):
            raise ValueError(f"`meta_data` must be a dict, got type {type(meta_data)}.")
        # make sure all elements in metadata are little endian
        meta_data = switch_endianness(meta_data, "<")

        meta_data[ImageMetaKey.FILENAME_OR_OBJ] = path
        img = MetaTensor.ensure_torch_and_prune_meta(
            img_array,
            meta_data,
            simple_keys=self.simple_keys,
            pattern=self.pattern,
            sep=self.sep,
        )
        if self.ensure_channel_first:
            img = monai.transforms.EnsureChannelFirst()(img)
        return img


class LoadNIfTIFromLocalCached(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
        dtype: DtypeLike | None = np.float32,
        meta_keys: KeysCollection | None = None,
        meta_key_postfix: str = PostFix.meta(),
        overwriting: bool = False,
        ensure_channel_first: bool = False,
        simple_keys: bool = False,
        prune_meta_pattern: str | None = None,
        prune_meta_sep: str = ".",
    ):
        super().__init__(keys, allow_missing_keys)
        self._loader = LoadNIfTIFromLocalCache(
            dtype=dtype,
            ensure_channel_first=ensure_channel_first,
            simple_keys=simple_keys,
            prune_meta_pattern=prune_meta_pattern,
            prune_meta_sep=prune_meta_sep,
        )
        if not isinstance(meta_key_postfix, str):
            raise TypeError(
                f"meta_key_postfix must be a str but is {type(meta_key_postfix).__name__}."
            )
        self.meta_keys = (
            ensure_tuple_rep(None, len(self.keys))
            if meta_keys is None
            else ensure_tuple(meta_keys)
        )
        if len(self.keys) != len(self.meta_keys):
            raise ValueError(
                f"meta_keys should have the same length as keys, got {len(self.keys)} and {len(self.meta_keys)}."
            )
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.overwriting = overwriting

    def __call__(self, data):
        d = dict(data)
        for key, meta_key, meta_key_postfix in self.key_iterator(
            d, self.meta_keys, self.meta_key_postfix
        ):
            data = self._loader(d[key])
            d[key] = data
        return d


class LoadNIfTIFromHFHub(Transform):
    def __init__(
        self,
        dtype: DtypeLike | None = np.float32,
        ensure_channel_first: bool = False,
        simple_keys: bool = False,
        prune_meta_pattern: str | None = None,
        prune_meta_sep: str = ".",
    ):
        self.dtype = dtype
        self.ensure_channel_first = ensure_channel_first
        self.simple_keys = simple_keys
        self.pattern = prune_meta_pattern
        self.sep = prune_meta_sep

        self.fs = HfFileSystem()
        self.reader = NibabelReader()

    def __call__(self, url: str):
        url = LoadNIfTIFromHFHub._convert_to_hffs_path(url)
        with self.fs.open(url, mode="rb") as f:
            img = nib.Nifti1Image.from_stream(
                IndexedGzipFile(fileobj=BytesIO(f.read()))
            )
        img = correct_nifti_header_if_necessary(img)
        img_array, meta_data = self.reader.get_data(img)
        img_array = convert_to_dst_type(img_array, dst=img_array, dtype=self.dtype)[0]
        if not isinstance(meta_data, dict):
            raise ValueError(f"`meta_data` must be a dict, got type {type(meta_data)}.")
        # make sure all elements in metadata are little endian
        meta_data = switch_endianness(meta_data, "<")

        meta_data[ImageMetaKey.FILENAME_OR_OBJ] = url
        img = MetaTensor.ensure_torch_and_prune_meta(
            img_array,
            meta_data,
            simple_keys=self.simple_keys,
            pattern=self.pattern,
            sep=self.sep,
        )
        if self.ensure_channel_first:
            img = monai.transforms.EnsureChannelFirst()(img)
        return img

    @staticmethod
    def _convert_to_hffs_path(url: str):
        if url.startswith("https://huggingface.co/datasets/"):
            parts = url.split("/")
            return f"hf://{'/'.join(parts[3:6])}/{'/'.join(parts[8:])}"
        return url


class LoadNIfTIFromHFHubd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
        dtype: DtypeLike | None = np.float32,
        meta_keys: KeysCollection | None = None,
        meta_key_postfix: str = PostFix.meta(),
        overwriting: bool = False,
        ensure_channel_first: bool = False,
        simple_keys: bool = False,
        prune_meta_pattern: str | None = None,
        prune_meta_sep: str = ".",
    ):
        super().__init__(keys, allow_missing_keys)
        self._loader = LoadNIfTIFromHFHub(
            dtype=dtype,
            ensure_channel_first=ensure_channel_first,
            simple_keys=simple_keys,
            prune_meta_pattern=prune_meta_pattern,
            prune_meta_sep=prune_meta_sep,
        )
        if not isinstance(meta_key_postfix, str):
            raise TypeError(
                f"meta_key_postfix must be a str but is {type(meta_key_postfix).__name__}."
            )
        self.meta_keys = (
            ensure_tuple_rep(None, len(self.keys))
            if meta_keys is None
            else ensure_tuple(meta_keys)
        )
        if len(self.keys) != len(self.meta_keys):
            raise ValueError(
                f"meta_keys should have the same length as keys, got {len(self.keys)} and {len(self.meta_keys)}."
            )
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.overwriting = overwriting

    def __call__(self, data):
        d = dict(data)
        for key, meta_key, meta_key_postfix in self.key_iterator(
            d, self.meta_keys, self.meta_key_postfix
        ):
            data = self._loader(d[key])
            d[key] = data
        return d


class UnifyUnusualDICOM(Transform):
    """
    Correct DICOM pixel_array if PixelRepresentation == 1 and BitsAllocated != BitsStored

    Steps:
    1. Convert data back to the original signed int16.
    2. Compute the number of bits to shift over (BitsShift = BitsAllocated - BitsStored)
    3. Left shift by BitsShift then right shift by BitsShift
    4. Convert data back to the default dtype for metatensor (float32)

    By default all dicom files in this dataset `rsna-2023-abdominal-trauma-detection` is in
    - uint16 if Pixel Representation = 0
    - int16 if Pixel Representation = 1
    Refer: https://dicom.innolitics.com/ciods/rt-dose/image-pixel/00280103

    Warning:
    - Use this transform on the test set as we expect to take DICOM series as input instead of NIfTI.
    - The passed in metatensor must have the following DICOM metadata:
        - Pixel Representation
        - Bits Allocated
        - Bits Stored
    - To have a metatensor that has those metadata:
        - Set reader to be PydicomReader with prune_metadata=False, i.e., monai.transforms.LoadImaged(..., reader=PydicomReader(prune_metadata=False))

    """

    def __init__(self):
        self.DCM_ATTR2TAG = {
            "Bits Allocated": "00280100",  # http://dicomlookup.com/lookup.asp?sw=Tnumber&q=(0028,0100)
            "Bits Stored": "00280101",  # http://dicomlookup.com/lookup.asp?sw=Tnumber&q=(0028,0101)
            "Pixel Representation": "00280103",  # http://dicomlookup.com/lookup.asp?sw=Tnumber&q=(0028,0103)
        }

    def __call__(self, data):
        if not all([dcm_tag in data.meta for dcm_tag in self.DCM_ATTR2TAG.values()]):
            raise Exception(
                f"Attribute tags of {self.DCM_ATTR2TAG} must exist in the dicom metadata to use this transform `{self.__class__.__name__}. Hint: Set reader to be PydicomReader with prune_metadata=False, i.e., monai.transforms.LoadImaged(..., reader=PydicomReader(prune_metadata=False))`"
            )
        pixel_representation = data.meta[self.DCM_ATTR2TAG["Pixel Representation"]][
            "Value"
        ][0]
        bits_allocated = data.meta[self.DCM_ATTR2TAG["Bits Allocated"]]["Value"][0]
        bits_stored = data.meta[self.DCM_ATTR2TAG["Bits Stored"]]["Value"][0]
        data = UnifyUnusualDICOM._standardize_dicom_pixels(
            data, pixel_representation, bits_allocated, bits_stored
        )
        return data

    @staticmethod
    def _standardize_dicom_pixels(
        data: torch.Tensor,
        pixel_representation: int,
        bits_allocated: int,
        bits_stored: int,
    ):
        bits_shift = bits_allocated - bits_stored

        if pixel_representation == 1 and bits_shift != 0:
            dtype_before = data.dtype
            dtype_shift = torch.int16
            data = data.to(dtype_shift)
            data = (data << bits_shift).to(dtype_shift) >> bits_shift
            data = data.to(dtype_before)
        return data


class UnifyUnusualDICOMd(MapTransform):
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self._unify_unusual_dicom = UnifyUnusualDICOM()

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            data = self._unify_unusual_dicom(d[key])
            d[key] = data
        return d


class UnifyUnusualNIfTI(Transform):
    """
    Correct NIfTI pixel values if PixelRepresentation == 1 and BitsAllocated != BitsStored.

    Steps:
    1. Convert data back to the original signed int16.
    2. Compute the number of bits to shift over (BitsShift = BitsAllocated - BitsStored)
    3. Left shift by BitsShift then right shift by BitsShift
    4. Convert data back to the default dtype for metatensor (float32)

    By default all dicom files in this dataset `rsna-2023-abdominal-trauma-detection` is in
    - uint16 if Pixel Representation = 0
    - int16 if Pixel Representation = 1
    Refer: https://dicom.innolitics.com/ciods/rt-dose/image-pixel/00280103

    Warning:
    - This transform only works for DICOM series that has been converted to NIfTI format and
    has a precomputed csv file that tracks the series that has unusual DICOM pixel representation format (`potential_unusual_dicom_series_meta.csv`).
    - This transform is not applicable for data that we have not preprocess yet (e.g. test set)
    - Use a different custom transform for test set (e.g. `UnifyUnusualDICOM`) as we expect to take DICOM series as input instead of NIfTI

    Why do we this?
    - NIfTI file doesn't store the Pixel Representation, Bits Allocated, and Bits Stored metadata.
    - The reason behind using a NIfTI file is to allow for easier data loading during training phase.

    """

    def __init__(
        self,
        x_key: str = "img",
        metadata_key: str = "metadata",
        meta_pixel_representation_key: str = "pixel_representation",
        meta_bits_allocated_key: str = "bits_allocated",
        meta_bits_stored_key: str = "bits_stored",
    ):
        self.x_key = x_key
        self.metadata_key = metadata_key
        self.pixel_representation_key = meta_pixel_representation_key
        self.bits_allocated_key = meta_bits_allocated_key
        self.bits_stored_key = meta_bits_stored_key

    def __call__(self, data):
        if not self.metadata_key in data or not self.x_key in data:
            raise KeyError(
                f"Key `{self.metadata_key}` of transform `{self.__class__.__name__}` was missing in the data."
            )

        if (
            not self.pixel_representation_key in data[self.metadata_key]
            or not self.bits_allocated_key in data[self.metadata_key]
            or not self.bits_stored_key in data[self.metadata_key]
        ):
            raise KeyError(
                f"Key `{self.pixel_representation_key}` or `{self.bits_allocated_key}` or `{self.bits_stored_key}` of transform `{self.__class__.__name__}` was missing in the metadata."
            )

        data[self.x_key] = UnifyUnusualNIfTI._standardize_dicom_pixels(
            data[self.x_key],
            data[self.metadata_key][self.pixel_representation_key],
            data[self.metadata_key][self.bits_allocated_key],
            data[self.metadata_key][self.bits_stored_key],
        )

        return data

    @staticmethod
    def _standardize_dicom_pixels(
        data: torch.Tensor,
        pixel_representation: int,
        bits_allocated: int,
        bits_stored: int,
    ):
        bits_shift = bits_allocated - bits_stored

        if pixel_representation == 1 and bits_shift != 0:
            dtype_before = data.dtype
            dtype_shift = torch.int16
            data = data.to(dtype_shift)
            data = (data << bits_shift).to(dtype_shift) >> bits_shift
            data = data.to(dtype_before)
        return data


def volume_transforms(
    crop_strategy: Optional[
        Literal["oversample", "center", "random", "none"]
    ] = "oversample",
    voxel_spacing: tuple[float, float, float] = (3.0, 3.0, 3.0),
    volume_size: tuple[int, int, int] = (96, 96, 96),
    axcodes: str = "RAS",
    streaming: bool = False,
) -> monai.transforms.Compose:
    if crop_strategy == "oversample":
        return monai.transforms.Compose(
            [
                LoadNIfTIFromHFHubd(keys=["img"])
                if streaming
                else LoadNIfTIFromLocalCached(keys=["img"]),
                monai.transforms.EnsureTyped(
                    keys=["img"], data_type="tensor", dtype=torch.float32
                ),
                UnifyUnusualNIfTI(
                    x_key="img",
                    metadata_key="metadata",
                    meta_pixel_representation_key="pixel_representation",
                    meta_bits_allocated_key="bits_allocated",
                    meta_bits_stored_key="bits_stored",
                ),
                monai.transforms.EnsureChannelFirstd(keys=["img"]),
                monai.transforms.Orientationd(keys=["img"], axcodes=axcodes),
                monai.transforms.Spacingd(
                    keys=["img"], pixdim=voxel_spacing, mode=["bilinear"]
                ),
                monai.transforms.NormalizeIntensityd(keys=["img"], nonzero=False),
                monai.transforms.ScaleIntensityd(keys=["img"], minv=-1.0, maxv=1.0),
                monai.transforms.SpatialPadd(keys=["img"], spatial_size=volume_size),
                monai.transforms.RandSpatialCropSamplesd(
                    keys=["img"],
                    roi_size=volume_size,
                    num_samples=3,
                    random_center=True,
                    random_size=False,
                ),
            ]
        )

    elif crop_strategy == "center":
        return monai.transforms.Compose(
            [
                LoadNIfTIFromHFHubd(keys=["img"])
                if streaming
                else LoadNIfTIFromLocalCached(keys=["img"]),
                monai.transforms.EnsureTyped(
                    keys=["img"], data_type="tensor", dtype=torch.float32
                ),
                UnifyUnusualNIfTI(
                    x_key="img",
                    metadata_key="metadata",
                    meta_pixel_representation_key="pixel_representation",
                    meta_bits_allocated_key="bits_allocated",
                    meta_bits_stored_key="bits_stored",
                ),
                monai.transforms.EnsureChannelFirstd(keys=["img"]),
                monai.transforms.Orientationd(keys=["img"], axcodes=axcodes),
                monai.transforms.Spacingd(
                    keys=["img"], pixdim=voxel_spacing, mode=["bilinear"]
                ),
                monai.transforms.NormalizeIntensityd(keys=["img"], nonzero=False),
                monai.transforms.ScaleIntensityd(keys=["img"], minv=-1.0, maxv=1.0),
                monai.transforms.SpatialPadd(keys=["img"], spatial_size=volume_size),
                monai.transforms.CenterSpatialCropd(keys=["img"], roi_size=volume_size),
            ]
        )

    elif crop_strategy == "random":
        return monai.transforms.Compose(
            [
                LoadNIfTIFromHFHubd(keys=["img"])
                if streaming
                else LoadNIfTIFromLocalCached(keys=["img"]),
                monai.transforms.EnsureTyped(
                    keys=["img"], data_type="tensor", dtype=torch.float32
                ),
                UnifyUnusualNIfTI(
                    x_key="img",
                    metadata_key="metadata",
                    meta_pixel_representation_key="pixel_representation",
                    meta_bits_allocated_key="bits_allocated",
                    meta_bits_stored_key="bits_stored",
                ),
                monai.transforms.EnsureChannelFirstd(keys=["img"]),
                monai.transforms.Orientationd(keys=["img"], axcodes=axcodes),
                monai.transforms.Spacingd(
                    keys=["img"], pixdim=voxel_spacing, mode=["bilinear"]
                ),
                monai.transforms.NormalizeIntensityd(keys=["img"], nonzero=False),
                monai.transforms.ScaleIntensityd(keys=["img"], minv=-1.0, maxv=1.0),
                monai.transforms.SpatialPadd(keys=["img"], spatial_size=volume_size),
                monai.transforms.RandSpatialCropd(
                    keys=["img"],
                    roi_size=volume_size,
                    random_center=True,
                    random_size=False,
                ),
            ]
        )

    elif crop_strategy == "none" or crop_strategy is None:
        return monai.transforms.Compose(
            [
                LoadNIfTIFromHFHubd(keys=["img"])
                if streaming
                else LoadNIfTIFromLocalCached(keys=["img"]),
                monai.transforms.EnsureTyped(
                    keys=["img"], data_type="tensor", dtype=torch.float32
                ),
                UnifyUnusualNIfTI(
                    x_key="img",
                    metadata_key="metadata",
                    meta_pixel_representation_key="pixel_representation",
                    meta_bits_allocated_key="bits_allocated",
                    meta_bits_stored_key="bits_stored",
                ),
                monai.transforms.EnsureChannelFirstd(keys=["img"]),
                monai.transforms.Orientationd(keys=["img"], axcodes=axcodes),
                monai.transforms.Spacingd(
                    keys=["img"], pixdim=voxel_spacing, mode=["bilinear"]
                ),
                monai.transforms.NormalizeIntensityd(keys=["img"], nonzero=False),
                monai.transforms.ScaleIntensityd(keys=["img"], minv=-1.0, maxv=1.0),
            ]
        )

    else:
        raise ValueError(
            f"crop_strategy must be one of ['oversample', 'center', 'random', 'none'], got {crop_strategy}."
        )


def mask_transforms(
    crop_strategy: Optional[Literal["oversample", "center", "none"]] = "oversample",
    voxel_spacing: tuple[float, float, float] = (3.0, 3.0, 3.0),
    volume_size: tuple[int, int, int] = (96, 96, 96),
    axcodes: str = "RAS",
    streaming: bool = False,
) -> monai.transforms.Compose:
    if crop_strategy == "oversample":
        return monai.transforms.Compose(
            [
                LoadNIfTIFromHFHubd(keys=["seg"])
                if streaming
                else LoadNIfTIFromLocalCached(keys=["seg"]),
                monai.transforms.EnsureTyped(
                    keys=["seg"], data_type="tensor", dtype=torch.float32
                ),
                monai.transforms.EnsureChannelFirstd(keys=["seg"]),
                monai.transforms.Orientationd(keys=["seg"], axcodes=axcodes),
                monai.transforms.Spacingd(
                    keys=["seg"], pixdim=voxel_spacing, mode=["nearest"]
                ),
                monai.transforms.SpatialPadd(keys=["seg"], spatial_size=volume_size),
                monai.transforms.RandSpatialCropSamplesd(
                    keys=["seg"],
                    roi_size=volume_size,
                    num_samples=3,
                    random_center=True,
                    random_size=False,
                ),
            ]
        )

    elif crop_strategy == "center":
        return monai.transforms.Compose(
            [
                LoadNIfTIFromHFHubd(keys=["seg"])
                if streaming
                else LoadNIfTIFromLocalCached(keys=["seg"]),
                monai.transforms.EnsureTyped(
                    keys=["seg"], data_type="tensor", dtype=torch.float32
                ),
                monai.transforms.EnsureChannelFirstd(keys=["seg"]),
                monai.transforms.Orientationd(keys=["seg"], axcodes=axcodes),
                monai.transforms.Spacingd(
                    keys=["seg"], pixdim=voxel_spacing, mode=["nearest"]
                ),
                monai.transforms.SpatialPadd(keys=["seg"], spatial_size=volume_size),
                monai.transforms.CenterSpatialCropd(keys=["seg"], roi_size=volume_size),
            ]
        )

    elif crop_strategy == "none" or crop_strategy is None:
        return monai.transforms.Compose(
            [
                LoadNIfTIFromHFHubd(keys=["seg"])
                if streaming
                else LoadNIfTIFromLocalCached(keys=["seg"]),
                monai.transforms.EnsureTyped(
                    keys=["seg"], data_type="tensor", dtype=torch.float32
                ),
                monai.transforms.EnsureChannelFirstd(keys=["seg"]),
                monai.transforms.Orientationd(keys=["seg"], axcodes=axcodes),
                monai.transforms.Spacingd(
                    keys=["seg"], pixdim=voxel_spacing, mode=["nearest"]
                ),
            ]
        )
    else:
        raise ValueError(
            f"crop_strategy must be one of ['oversample', 'center', 'none'], got {crop_strategy}."
        )


def slice_transforms(
    crop_strategy: Literal["ten", "five", "center", "random"] = "ten",
    shorter_edge_length: int = 256,
    slice_size: tuple[int, int] = (224, 224),
) -> torchvision.transforms.Compose:
    if crop_strategy == "ten":
        return torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=shorter_edge_length, antialias=True),
                torchvision.transforms.TenCrop(size=slice_size),
            ]
        )

    elif crop_strategy == "five":
        return torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=shorter_edge_length, antialias=True),
                torchvision.transforms.FiveCrop(size=slice_size),
            ]
        )

    elif crop_strategy == "center":
        return torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=shorter_edge_length, antialias=True),
                torchvision.transforms.CenterCrop(size=slice_size),
            ]
        )
    elif crop_strategy == "random":
        return torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=shorter_edge_length, antialias=True),
                torchvision.transforms.RandomCrop(size=slice_size),
            ]
        )

    else:
        raise ValueError(
            f"crop_strategy must be one of ['ten', 'five', 'center', 'random'], got {crop_strategy}."
        )
