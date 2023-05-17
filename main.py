from load_dataset import load_dataset
import pandas as pd

def main():
    data = load_dataset("dataset")
    mlmp_df = pd.DataFrame(data,columns=['song_file', 'interpreter', 'label','audio_type','mfccs'])
    print(mlmp_df)

if __name__ == "__main__":
    main()