import os
import cv2
from tqdm import tqdm

def crop_center_height_ratio_cv2(img_cv2, upper_ratio, lower_ratio):
    """
    OpenCV 이미지 ndarray를 받아 비율 기반 height 중앙 크롭 후 반환
    """
    h, w = img_cv2.shape[:2]

    center_y = h // 2
    start_y = int(center_y - upper_ratio * h)
    end_y = int(center_y + lower_ratio * h)

    start_y = max(start_y, 0)
    end_y = min(end_y, h)

    cropped_img = img_cv2[start_y:end_y, 0:w]
    return cropped_img


def crop_and_save_dataset(root_dir,out_dir, upper_ratio=0.3, lower_ratio=0.1):
    """
    root_dir 하위의 클래스별 폴더에서 이미지 읽고,
    크롭 후 '_crop'이 붙은 새 폴더에 저장
    """
    classes = [
        d for d in sorted(os.listdir(root_dir))
        if os.path.isdir(os.path.join(root_dir, d))
    ]

    for class_name in classes:
        src_class_dir = os.path.join(root_dir, class_name)
        dst_class_dir = os.path.join(out_dir, f"{class_name}")
        os.makedirs(dst_class_dir, exist_ok=True)

        img_files = [f for f in os.listdir(src_class_dir) if f.lower().endswith('.jpg')]

        print(f"Processing class '{class_name}', {len(img_files)} images...")

        for img_file in tqdm(img_files):
            src_path = os.path.join(src_class_dir, img_file)
            dst_path = os.path.join(dst_class_dir, img_file)

            img_cv2 = cv2.imread(src_path)
            if img_cv2 is None:
                print(f"Warning: Cannot read {src_path}")
                continue

            cropped_img = crop_center_height_ratio_cv2(img_cv2, upper_ratio, lower_ratio)
            cv2.imwrite(dst_path, cropped_img)

    print("Cropping and saving completed.")

root_dir = "../data/train"  # 실제 경로로 변경하세요
out_dir = '../data/train_crop'
crop_and_save_dataset(root_dir, out_dir=out_dir, upper_ratio=0.3, lower_ratio=0.1)