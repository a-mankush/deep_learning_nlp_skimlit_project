import os
import shutil

import gdown
import tensorflow as tf


def download_folder_from_google_drive(folder_id, output_path):
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    gdown.download_folder(url, output=output_path, quiet=False, use_cookies=False)


def download_model():
    # The ID of the shared Google Drive folder (skimLit_8b)
    folder_id = (
        "1s4YLSBbFv9BP4FhcICObZI4xP6KSb3xc"  # Replace this with the actual folder ID
    )
    output_path = "skimLit_8b"

    # Download the model folder
    print(f"Downloading {output_path} folder...")
    try:
        download_folder_from_google_drive(folder_id, output_path)
    except Exception as e:
        print(f"Error downloading folder: {e}")
        return

    # Check if the folder was downloaded and has content
    if not os.path.exists(output_path) or not os.listdir(output_path):
        print(
            f"Error: The downloaded folder '{output_path}' is empty or does not exist."
        )
        return

    # # Load the model
    # print("Loading the model...")
    # try:
    #     model = tf.saved_model.load(output_path)
    #     print("Model loaded successfully!")
    #     return model
    # except Exception as e:
    #     print(f"Error loading the model: {e}")
    #     return

    # Optional: Clean up downloaded files
    # Uncomment the following line if want to remove the downloaded folder after loading the model
    # shutil.rmtree("skimLit_8b")
