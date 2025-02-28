import os
import shutil
from sklearn.model_selection import train_test_split

def split_dataset():
    """Split the dataset into train, validation, and test sets."""
    base_dir = 'dataset'
    labels_dir = os.path.join(base_dir, 'labels', 'labels_native')
    images_dir = os.path.join(base_dir, 'images', 'native')
    train_images_dir = os.path.join(base_dir, 'images', 'train')
    val_images_dir = os.path.join(base_dir, 'images', 'val')
    test_images_dir = os.path.join(base_dir, 'images', 'test')
    train_labels_dir = os.path.join(base_dir, 'labels', 'train')
    val_labels_dir = os.path.join(base_dir, 'labels', 'val')
    test_labels_dir = os.path.join(base_dir, 'labels', 'test')

    # Create directories if they don't exist
    for dir_path in [train_images_dir, val_images_dir, test_images_dir, train_labels_dir, val_labels_dir, test_labels_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # Get list of label files and corresponding images
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    image_files = [f.replace('.txt', '.tif') for f in label_files]

    # Split dataset
    train_val_files, test_files = train_test_split(label_files, test_size=0.2, random_state=42)
    train_files, val_files = train_test_split(train_val_files, test_size=0.1, random_state=42)

    def copy_files(file_list, src_images_dir, src_labels_dir, dst_images_dir, dst_labels_dir):
        """Copy image and label files to destination directories."""
        for file in file_list:
            shutil.copy(os.path.join(src_labels_dir, file), os.path.join(dst_labels_dir, file))
            image_file = file.replace('.txt', '.tif')
            shutil.copy(os.path.join(src_images_dir, image_file), os.path.join(dst_images_dir, image_file))

    # Copy files to respective directories
    copy_files(train_files, images_dir, labels_dir, train_images_dir, train_labels_dir)
    copy_files(val_files, images_dir, labels_dir, val_images_dir, val_labels_dir)
    copy_files(test_files, images_dir, labels_dir, test_images_dir, test_labels_dir)

    print(f"Dataset split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

if __name__ == "__main__":
    split_dataset()