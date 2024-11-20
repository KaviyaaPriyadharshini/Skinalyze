import os

def file_with_labels(root_dir,output_file):
# Open the output text file in write mode
    with open(output_file, 'w') as f:
        # Traverse through the skin type folders (dry, normal, oily)
        for skin_type in ['dry', 'normal', 'oily']:
            skin_type_folder = os.path.join(root_dir, skin_type)
        
            # Check if the folder exists
            if os.path.exists(skin_type_folder):
                # Loop through each image in the skin type folder
                for image_name in os.listdir(skin_type_folder):
                    image_path = os.path.join(skin_type_folder, image_name)
                
                    # Check if the current file is an image (optional)
                    if image_name.endswith(('.png', '.jpg', '.jpeg')):
                        # Write the image path and label to the output file
                        f.write(f"{image_path}\t{skin_type}\n")


# Define the root directory where your skin type folders are located
test_dir = "/Users/daksha/Desktop/kavs/Oily-Dry-Skin-Types/test"
train_dir = "/Users/daksha/Desktop/kavs/Oily-Dry-Skin-Types/train"
valid_dir = "/Users/daksha/Desktop/kavs/Oily-Dry-Skin-Types/valid"
train_output = "train.txt"
test_output = "test.txt"
valid_output = "valid.txt"
file_with_labels(train_dir,train_output)
file_with_labels(test_dir,test_output)
file_with_labels(valid_dir,valid_output)
