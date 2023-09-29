import datasets

if __name__ == "__main__":
    clf_ds = datasets.load_dataset(
        "jherng/rsna-2023-abdominal-trauma-detection", "classification", num_proc=8
    )
    masked_clf_ds = datasets.load_dataset(
        "jherng/rsna-2023-abdominal-trauma-detection", "classification-with-mask", num_proc=8
    )
    seg_ds = datasets.load_dataset(
        "jherng/rsna-2023-abdominal-trauma-detection", "segmentation", num_proc=8
    )
    print(clf_ds)
    print(masked_clf_ds)
    print(seg_ds)
