import urllib
import numpy as np
import pandas as pd
import datasets
from sklearn.model_selection import train_test_split


_CITATION = """\
@InProceedings{huggingface:dataset,
title = {RSNA 2023 Abdominal Trauma Detection Dataset},
author={Hong Jia Herng},
year={2023}
}
@misc{rsna-2023-abdominal-trauma-detection,
    author = {Errol Colak, Hui-Ming Lin, Robyn Ball, Melissa Davis, Adam Flanders, Sabeena Jalal, Kirti Magudia, Brett Marinelli, Savvas Nicolaou, Luciano Prevedello, Jeff Rudie, George Shih, Maryam Vazirabad, John Mongan},
    title = {RSNA 2023 Abdominal Trauma Detection},
    publisher = {Kaggle},
    year = {2023},
    url = {https://kaggle.com/competitions/rsna-2023-abdominal-trauma-detection}
}
"""

_DESCRIPTION = """\
This dataset is the preprocessed version of the dataset from RSNA 2023 Abdominal Trauma Detection Kaggle Competition. 
It is tailored for segmentation and classification tasks. It contains 3 different configs as described below:
- segmentation: 206 instances where each instance includes a CT scan in NIfTI format, a segmentation mask in NIfTI format, and its relevant metadata (e.g., patient_id, series_id, incomplete_organ, aortic_hu, pixel_representation, bits_allocated, bits_stored)
- classification: 4711 instances where each instance includes a CT scan in NIfTI format, target labels (e.g., extravasation, bowel, kidney, liver, spleen, any_injury), and its relevant metadata (e.g., patient_id, series_id, incomplete_organ, aortic_hu, pixel_representation, bits_allocated, bits_stored)
- classification-with-mask: 206 instances where each instance includes a CT scan in NIfTI format, a segmentation mask in NIfTI format, target labels (e.g., extravasation, bowel, kidney, liver, spleen, any_injury), and its relevant metadata (e.g., patient_id, series_id, incomplete_organ, aortic_hu, pixel_representation, bits_allocated, bits_stored)

All CT scans and segmentation masks had already been resampled with voxel spacing (2.0, 2.0, 3.0) and thus its reduced file size.
"""

_NAME = "rsna-2023-abdominal-trauma-detection"

_HOMEPAGE = f"https://huggingface.co/datasets/jherng/{_NAME}"

_LICENSE = "MIT"

_URL = f"https://huggingface.co/datasets/jherng/{_NAME}/resolve/main/"


class RSNA2023AbdominalTraumaDetectionConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        self.test_size = kwargs.pop("test_size", 0.1)
        self.random_state = kwargs.pop("random_state", 42)
        super(RSNA2023AbdominalTraumaDetectionConfig, self).__init__(**kwargs)


class RSNA2023AbdominalTraumaDetection(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        RSNA2023AbdominalTraumaDetectionConfig(
            name="segmentation",
            version=VERSION,
            description="This part of the dataset loads the CT scans, segmentation masks, and metadata.",
        ),
        RSNA2023AbdominalTraumaDetectionConfig(
            name="classification",
            version=VERSION,
            description="This part of the dataset loads the CT scans, target labels, and metadata.",
        ),
        RSNA2023AbdominalTraumaDetectionConfig(
            name="classification-with-mask",
            version=VERSION,
            description="This part of the dataset loads the CT scans, segmentation masks, target labels, and metadata.",
        ),
    ]

    DEFAULT_CONFIG_NAME = "classification"  # It's not mandatory to have a default configuration. Just use one if it make sense.
    BUILDER_CONFIG_CLASS = RSNA2023AbdominalTraumaDetectionConfig

    def _info(self):
        if self.config.name == "segmentation":
            features = datasets.Features(
                {
                    "img_path": datasets.Value("string"),
                    "seg_path": datasets.Value("string"),
                    "metadata": {
                        "series_id": datasets.Value("int32"),
                        "patient_id": datasets.Value("int32"),
                        "incomplete_organ": datasets.Value("bool"),
                        "aortic_hu": datasets.Value("float32"),
                        "pixel_representation": datasets.Value("int32"),
                        "bits_allocated": datasets.Value("int32"),
                        "bits_stored": datasets.Value("int32"),
                    },
                }
            )
        elif self.config.name == "classification-with-mask":
            features = datasets.Features(
                {
                    "img_path": datasets.Value("string"),
                    "seg_path": datasets.Value("string"),
                    "bowel": datasets.ClassLabel(
                        num_classes=2, names=["healthy", "injury"]
                    ),
                    "extravasation": datasets.ClassLabel(
                        num_classes=2, names=["healthy", "injury"]
                    ),
                    "kidney": datasets.ClassLabel(
                        num_classes=3, names=["healthy", "low", "high"]
                    ),
                    "liver": datasets.ClassLabel(
                        num_classes=3, names=["healthy", "low", "high"]
                    ),
                    "spleen": datasets.ClassLabel(
                        num_classes=3, names=["healthy", "low", "high"]
                    ),
                    "any_injury": datasets.Value("bool"),
                    "metadata": {
                        "series_id": datasets.Value("int32"),
                        "patient_id": datasets.Value("int32"),
                        "incomplete_organ": datasets.Value("bool"),
                        "aortic_hu": datasets.Value("float32"),
                        "pixel_representation": datasets.Value("int32"),
                        "bits_allocated": datasets.Value("int32"),
                        "bits_stored": datasets.Value("int32"),
                    },
                }
            )

        else:
            features = datasets.Features(
                {
                    "img_path": datasets.Value("string"),
                    "bowel": datasets.ClassLabel(
                        num_classes=2, names=["healthy", "injury"]
                    ),
                    "extravasation": datasets.ClassLabel(
                        num_classes=2, names=["healthy", "injury"]
                    ),
                    "kidney": datasets.ClassLabel(
                        num_classes=3, names=["healthy", "low", "high"]
                    ),
                    "liver": datasets.ClassLabel(
                        num_classes=3, names=["healthy", "low", "high"]
                    ),
                    "spleen": datasets.ClassLabel(
                        num_classes=3, names=["healthy", "low", "high"]
                    ),
                    "any_injury": datasets.Value("bool"),
                    "metadata": {
                        "series_id": datasets.Value("int32"),
                        "patient_id": datasets.Value("int32"),
                        "incomplete_organ": datasets.Value("bool"),
                        "aortic_hu": datasets.Value("float32"),
                        "pixel_representation": datasets.Value("int32"),
                        "bits_allocated": datasets.Value("int32"),
                        "bits_stored": datasets.Value("int32"),
                    },
                }
            )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # segmentation: 206 segmentations and the relevant imgs, train_series_meta.csv, train_dicom_tags.parquet
        # classification: 4711 all imgs, train.csv, train_series_meta.csv, train_dicom_tags.parquet
        # classification-with-mask: 206 segmentations and the relevant imgs, train.csv, train_series_meta.csv, train_dicom_tags.parquet
        series_meta_df = pd.read_csv(
            dl_manager.download_and_extract(
                urllib.parse.urljoin(_URL, "train_series_meta.csv")
            )
        )
        series_meta_df["img_download_url"] = series_meta_df.apply(
            lambda x: urllib.parse.urljoin(
                _URL,
                f"train_images/{int(x['patient_id'])}/{int(x['series_id'])}.nii.gz",
            ),
            axis=1,
        )
        series_meta_df["seg_download_url"] = series_meta_df.apply(
            lambda x: urllib.parse.urljoin(
                _URL, f"segmentations/{int(x['series_id'])}.nii.gz"
            ),
            axis=1,
        )
        if (
            self.config.name == "classification-with-mask"
            or self.config.name == "segmentation"
        ):
            series_meta_df = series_meta_df.loc[series_meta_df["has_segmentation"] == 1]

            series_meta_df["img_cache_path"] = dl_manager.download(
                series_meta_df["img_download_url"].tolist()
            )
            series_meta_df["seg_cache_path"] = dl_manager.download(
                series_meta_df["seg_download_url"].tolist()
            )

        else:
            series_meta_df["img_cache_path"] = dl_manager.download(
                series_meta_df["img_download_url"].tolist()
            )
            series_meta_df["seg_cache_path"] = None

        dicom_tags_df = datasets.load_dataset(
            "parquet", data_files=urllib.parse.urljoin(_URL, "train_dicom_tags.parquet")
        )["train"].to_pandas()[
            [
                "SeriesInstanceUID",
                "PixelRepresentation",
                "BitsAllocated",
                "BitsStored",
            ]
        ]
        dicom_tags_df["SeriesID"] = dicom_tags_df["SeriesInstanceUID"].apply(
            lambda x: int(x.split(".")[-1])
        )
        dicom_tags_df = dicom_tags_df.drop(labels=["SeriesInstanceUID"], axis=1)
        dicom_tags_df = dicom_tags_df.groupby(by=["SeriesID"], as_index=False).first()
        dicom_tags_df = dicom_tags_df.rename(
            columns={
                "SeriesID": "series_id",
                "PixelRepresentation": "pixel_representation",
                "BitsAllocated": "bits_allocated",
                "BitsStored": "bits_stored",
            }
        )
        series_meta_df = pd.merge(
            left=series_meta_df, right=dicom_tags_df, how="inner", on="series_id"
        )

        self.labels_df = (
            pd.read_csv(
                dl_manager.download_and_extract(urllib.parse.urljoin(_URL, "train.csv"))
            )
            if self.config.name != "segmentation"
            else None
        )

        train_series_meta_df, test_series_meta_df = train_test_split(
            series_meta_df, test_size=self.config.test_size, random_state=self.config.random_state, shuffle=True
        )

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepaths": train_series_meta_df[
                        [
                            "series_id",
                            "patient_id",
                            "img_cache_path",
                            "seg_cache_path",
                            "incomplete_organ",
                            "aortic_hu",
                            "pixel_representation",
                            "bits_allocated",
                            "bits_stored",
                        ]
                    ].to_dict("records"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepaths": test_series_meta_df[
                        [
                            "series_id",
                            "patient_id",
                            "img_cache_path",
                            "seg_cache_path",
                            "incomplete_organ",
                            "aortic_hu",
                            "pixel_representation",
                            "bits_allocated",
                            "bits_stored",
                        ]
                    ].to_dict("records"),
                },
            ),
        ]

    def _generate_examples(
        self,
        filepaths,
    ):
        if self.config.name == "segmentation":
            for key, series_meta in enumerate(filepaths):
                yield key, {
                    "img_path": series_meta["img_cache_path"],
                    "seg_path": series_meta["seg_cache_path"],
                    "metadata": {
                        "series_id": series_meta["series_id"],
                        "patient_id": series_meta["patient_id"],
                        "incomplete_organ": series_meta["incomplete_organ"],
                        "aortic_hu": series_meta["aortic_hu"],
                        "pixel_representation": series_meta["pixel_representation"],
                        "bits_allocated": series_meta["bits_allocated"],
                        "bits_stored": series_meta["bits_stored"],
                    },
                }

        elif self.config.name == "classification-with-mask":
            for key, series_meta in enumerate(filepaths):
                label_data = (
                    self.labels_df.loc[
                        self.labels_df["patient_id"] == series_meta["patient_id"]
                    ]
                    .iloc[0]
                    .to_dict()
                )

                yield key, {
                    "img_path": series_meta["img_cache_path"],
                    "seg_path": series_meta["seg_cache_path"],
                    "bowel": np.argmax(
                        [label_data["bowel_healthy"], label_data["bowel_injury"]]
                    ),
                    "extravasation": np.argmax(
                        [
                            label_data["extravasation_healthy"],
                            label_data["extravasation_injury"],
                        ]
                    ),
                    "kidney": np.argmax(
                        [
                            label_data["kidney_healthy"],
                            label_data["kidney_low"],
                            label_data["kidney_high"],
                        ]
                    ),
                    "liver": np.argmax(
                        [
                            label_data["liver_healthy"],
                            label_data["liver_low"],
                            label_data["liver_high"],
                        ]
                    ),
                    "spleen": np.argmax(
                        [
                            label_data["spleen_healthy"],
                            label_data["spleen_low"],
                            label_data["spleen_high"],
                        ]
                    ),
                    "any_injury": label_data["any_injury"],
                    "metadata": {
                        "series_id": series_meta["series_id"],
                        "patient_id": series_meta["patient_id"],
                        "incomplete_organ": series_meta["incomplete_organ"],
                        "aortic_hu": series_meta["aortic_hu"],
                        "pixel_representation": series_meta["pixel_representation"],
                        "bits_allocated": series_meta["bits_allocated"],
                        "bits_stored": series_meta["bits_stored"],
                    },
                }
        else:
            for key, series_meta in enumerate(filepaths):
                label_data = (
                    self.labels_df.loc[
                        self.labels_df["patient_id"] == series_meta["patient_id"]
                    ]
                    .iloc[0]
                    .to_dict()
                )

                yield key, {
                    "img_path": series_meta["img_cache_path"],
                    "bowel": np.argmax(
                        [label_data["bowel_healthy"], label_data["bowel_injury"]]
                    ),
                    "extravasation": np.argmax(
                        [
                            label_data["extravasation_healthy"],
                            label_data["extravasation_injury"],
                        ]
                    ),
                    "kidney": np.argmax(
                        [
                            label_data["kidney_healthy"],
                            label_data["kidney_low"],
                            label_data["kidney_high"],
                        ]
                    ),
                    "liver": np.argmax(
                        [
                            label_data["liver_healthy"],
                            label_data["liver_low"],
                            label_data["liver_high"],
                        ]
                    ),
                    "spleen": np.argmax(
                        [
                            label_data["spleen_healthy"],
                            label_data["spleen_low"],
                            label_data["spleen_high"],
                        ]
                    ),
                    "any_injury": label_data["any_injury"],
                    "metadata": {
                        "series_id": series_meta["series_id"],
                        "patient_id": series_meta["patient_id"],
                        "incomplete_organ": series_meta["incomplete_organ"],
                        "aortic_hu": series_meta["aortic_hu"],
                        "pixel_representation": series_meta["pixel_representation"],
                        "bits_allocated": series_meta["bits_allocated"],
                        "bits_stored": series_meta["bits_stored"],
                    },
                }
