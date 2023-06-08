import requests
import os


def download_txt(url: str, folder_path: str, filename: str):
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Create the full file path
    file_path = os.path.join(folder_path, filename)

    # Download the file and save it to the folder
    response = requests.get(url)
    response.raise_for_status()  # Check for any errors during the download

    with open(file_path, "wb") as file:
        file.write(response.content)

    print(f"The file has been downloaded and saved to {file_path}.")
