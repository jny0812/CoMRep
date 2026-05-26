import os
import pandas as pd
import cv2
import torch

# Fovea 좌표 CSV
df = pd.read_csv(r'dataset/ADAM/Validation/Fovea_location.csv')

IMG_FOLDER = r'dataset/ADAM/Validation/image'
OUT_MACULA = r'dataset/ADAM/Validation/macula-crops-35pct'
os.makedirs(OUT_MACULA, exist_ok=True)

for _, row in df.iterrows():
    img_name = row['imgName']
    x_center = int(row['Fovea_X'])
    y_center = int(row['Fovea_Y'])

    img_path = os.path.join(IMG_FOLDER, img_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ 파일을 읽을 수 없습니다: {img_path}")
        continue

    h, w = img.shape[:2]
    # 1) crop_size를 39% 비율로 계산
    crop_size = int(min(h, w) * 0.35)
    half = crop_size // 2

    # 2) 크롭 박스 좌표 (image 경계 내로 클리핑)
    x1 = max(x_center - half, 0)
    y1 = max(y_center - half, 0)
    x2 = min(x_center + half, w)
    y2 = min(y_center + half, h)

    macula = img[y1:y2, x1:x2]

    # 3) 크롭 결과가 정사각형이 아니면 패딩 또는 리사이즈
    if macula.shape[0] != crop_size or macula.shape[1] != crop_size:
        # 부족한 픽셀 계산
        top_pad = 0 if y1 > 0 else (half - y_center)
        left_pad = 0 if x1 > 0 else (half - x_center)
        bottom_pad = 0 if y2 < h else (y_center + half - h)
        right_pad = 0 if x2 < w else (x_center + half - w)

        macula = cv2.copyMakeBorder(
            macula,
            top=top_pad,
            bottom=bottom_pad,
            left=left_pad,
            right=right_pad,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )
        # 다시 정확한 사이즈로 리사이즈
        macula = cv2.resize(macula, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)

    # 4) 저장 (원본 이름 유지)
    out_path = os.path.join(OUT_MACULA, img_name)
    cv2.imwrite(out_path, macula)