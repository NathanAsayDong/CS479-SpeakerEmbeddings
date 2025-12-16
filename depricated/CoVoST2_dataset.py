from datasets import DownloadConfig, load_dataset
import os

class CoVoST2Dataset():
    """
    Dataset wraper for the CoVoST2 dataset. The experiments focus on rows where the client_id column is more than 1 instance.
    """
    def __init__(self, source_language: str, target_language: str):
        self.DATASET_DIR = "data/covost2"
        os.makedirs(self.DATASET_DIR, exist_ok=True)

        print(f"Downloading Common Voice 4.0 English data to be used by CoVoST2...")
        download_config = DownloadConfig(
            cache_dir=self.DATASET_DIR
        )
        self.dataset = load_dataset("facebook/covost2")


if __name__ == "__main__":
    dataset = CoVoST2Dataset(source_language="en", target_language="es")
    print(dataset[0])

    
        