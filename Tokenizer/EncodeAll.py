import os
from Encoder import our_encoder

# Path to the target folder
folder_path = r"C:\path-to-your-audio-files-folder"
save_path = r"C:\path-to-the-results-tokens-folder\token-"

# Iterate over all files in the folder (including subfolders)
for root, dirs, files in os.walk(folder_path):
    for file in files:
        # Construct the full path to the file
        file_path = os.path.join(root, file)

        # Construct the output file path + reduse the ".wav"\".mp3"
        output_file_path = save_path + file[:-4]

        # Check if the output file already exists
        if not os.path.exists(output_file_path):
            # Pass the file name to the function
            our_encoder(file_path, output_file_path)
        else:
            print(f"File {output_file_path} already exists, skipping...")

