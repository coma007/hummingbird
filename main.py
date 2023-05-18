import glob
from load_dataset import load_dataset
from prepare_dataset import prepare_dataset


def main():
    data = load_dataset("small_dataset")
    # print(data["audio_type"].unique())
    filepaths = []
    for label in (data["label"].unique()):
        filepaths += sorted(glob.glob(f"small_dataset/*{label}*/*.wav"))
    prepare_dataset(data, filepaths)
    # print(filepaths)


if __name__ == "__main__":
    main()
