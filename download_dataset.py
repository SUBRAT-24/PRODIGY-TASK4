import os
import kagglehub
import zipfile
import shutil

def download_dataset():
    print("Downloading Leap Gesture Recognition dataset...")
    
    # Create data directory if it doesn't exist
    data_dir = "leapGestRecog"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    try:
        # Download the dataset using kagglehub
        print("This may take a few minutes depending on your internet connection...")
        zip_path = kagglehub.dataset_download("gti-upm/leapgestrecog")
        
        # Extract the dataset
        print(f"Extracting dataset from {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        
        # The dataset might be in a subdirectory, move files if needed
        extracted_dir = os.path.join(os.path.dirname(zip_path), "leapgestrecog")
        if os.path.exists(extracted_dir):
            # Move all files from the extracted directory to our target directory
            for item in os.listdir(extracted_dir):
                s = os.path.join(extracted_dir, item)
                d = os.path.join(data_dir, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)
            
            # Clean up the extracted directory
            shutil.rmtree(os.path.dirname(zip_path))
        
        print(f"\nDataset successfully downloaded and extracted to: {os.path.abspath(data_dir)}")
        print("You can now run 'python hand_gesture_recognition.py' to train the model.")
        
    except Exception as e:
        print(f"An error occurred while downloading the dataset: {str(e)}")
        print("\nAlternative download options:")
        print("1. Download manually from: https://www.kaggle.com/gti-upm/leapgestrecog")
        print(f"2. Extract the downloaded zip file to: {os.path.abspath(data_dir)}")

if __name__ == "__main__":
    download_dataset()
