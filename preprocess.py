
import cv2
import argparse
import numpy as np
import os
from src.img_utils.foreground_extractor import *


def parse_args():
    parser = argparse.ArgumentParser(description="Background Extraction")
    parser.add_argument("-p", "--path", required=True, type=str,
                        help="path to folder dataset")
    parser.add_argument("-a", "--algo", default="KNN", type=str, help="Algorithm for background substraction")
    parser.add_argument("-d", "--dest", required=True, type=str, help="path to destination folder")
    args = parser.parse_args()
    return args


# TODO: FIx cannot substract background
# def extract_background(img: np.array, backsub):
#     fg_mask = backsub.apply(img)
#     return cv2.bitwise_and(img, img, mask=fg_mask)


def prepare_dataset(dataset_path, dest_path, algo="KNN"):
    backsub = cv2.createBackgroundSubtractorKNN() if algo == "KNN" else cv2.createBackgroundSubtractorMOG2()
    for subset_dir in os.listdir(dataset_path):
        # print(subset_dir)
        full_subset_dir_path = os.path.join(dataset_path, subset_dir)
        # print(full_subset_dir_path)
        for label_dir_path in os.listdir(full_subset_dir_path):
            full_label_dir_path = os.path.join(full_subset_dir_path, label_dir_path)
            for img_path in os.listdir(full_label_dir_path):
                # print(os.path.join(full_label_dir_path, img_path))
                img = cv2.imread(os.path.join(full_label_dir_path, img_path))
                # print(img)
                # img_substract = extract_foreground_contour(img)
                img_substract = extract_foreground_morphology(img,k_size=(4, 4))
                # print(img_substract)
                dest_dir = os.path.join(dest_path, full_label_dir_path)
                os.makedirs(dest_dir,exist_ok=True)
                save_path = os.path.join(dest_dir, img_path)
                print(save_path)
                cv2.imwrite(save_path, img_substract)


if __name__ == "__main__":
    args = parse_args()
    prepare_dataset(args.path, args.dest, args.algo)

# TODO: Get foreground based on opencv
# TODO: Substrackk using AND operator to get substrack 
