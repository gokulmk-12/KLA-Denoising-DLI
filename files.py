import os
import re
import shutil

def copy_images(source_degraded, source_gt, source_mask, dest_data, dest_labels, dest_masks):
    start_index = get_highest_index(dest_data)
    for root, _, files in os.walk(source_degraded):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                degraded_image_path = os.path.join(root, file)

                relative_path = os.path.relpath(root, source_degraded)
                gt_image_path = os.path.join(source_gt, relative_path, file)

                mask_image_filename = f"{os.path.splitext(file)[0]}_mask{os.path.splitext(file)[1]}"
                mask_image_path = os.path.join(source_mask, relative_path, mask_image_filename)
                
                new_data_filename = f"{start_index:03d}{os.path.splitext(file)[1]}"
                new_label_filename = f"{start_index:03d}{os.path.splitext(file)[1]}"
                new_mask_filename = f"{start_index:03d}{os.path.splitext(file)[1]}"

                shutil.copy(degraded_image_path, os.path.join(dest_data, new_data_filename))
                if os.path.exists(gt_image_path):
                    shutil.copy(gt_image_path, os.path.join(dest_labels, new_label_filename))
                if os.path.exists(mask_image_path):
                    shutil.copy(mask_image_path, os.path.join(dest_masks, new_mask_filename))
                
                start_index += 1

def get_highest_index(directory):
    max_index = -1
    for file in os.listdir(directory):
        if file.endswith(('.png', '.jpg', '.jpeg')):
            match = re.match(r'(\d+)', file)
            if match:
                index = int(match.group(1))
                if index > max_index:
                    max_index = index
    return max_index + 1 

def process_dataset(base_dir, dest_data_dir, dest_labels_dir, dest_masks_dir, split):
    os.makedirs(dest_data_dir, exist_ok=True)
    os.makedirs(dest_labels_dir, exist_ok=True)
    os.makedirs(dest_masks_dir, exist_ok=True)

    for obj in os.listdir(base_dir):
        if obj == split:
            obj_path = os.path.join(base_dir, obj)
            if os.path.isdir(obj_path):
                degraded_image_path = os.path.join(obj_path, 'Degraded_image')
                gt_image_path = os.path.join(obj_path, 'GT_clean_image')
                mask_image_path = os.path.join(obj_path, 'Defect_mask')

                if os.path.exists(degraded_image_path) and os.path.exists(gt_image_path) and os.path.exists(mask_image_path):
                    copy_images(degraded_image_path, gt_image_path, mask_image_path, dest_data_dir, dest_labels_dir, dest_masks_dir)


if __name__ == "__main__":
    base_dataset_dir = 'Denoising_Dataset_train_val/zipper' ## Change Class Here

    ## Change Dataset Location Here 
    
    train_data_dest = 'Dataset/data/Train'
    train_labels_dest = 'Dataset/label/Train'
    train_masks_dest = 'Dataset/mask/Train'

    val_data_dest = 'Dataset/data/Val'
    val_labels_dest = 'Dataset/label/Val'
    val_masks_dest = 'Dataset/mask/Val'

    process_dataset(base_dataset_dir, train_data_dest, train_labels_dest, train_masks_dest, 'Train')
    process_dataset(base_dataset_dir, val_data_dest, val_labels_dest, val_masks_dest, 'Val')
