# -*- coding: utf-8 -*-
# @Time: 2019-10-16 11:39
# @Author: Trible

import os
from PIL import Image
import numpy as np
from tool import utils
import traceback

anno_src = r"F:\CelebA\list_bbox_celeba.txt"
img_dir = r"F:\img_celeba"

save_path = r"F:\celeba1"

for face_size in [48, 12, 24]:

    print("gen %i image" % face_size)
    # 样本图片存储路径
    positive_image_dir = os.path.join(save_path, str(face_size), "positive")
    negative_image_dir = os.path.join(save_path, str(face_size), "negative")
    part_image_dir = os.path.join(save_path, str(face_size), "part")

    for dir_path in [positive_image_dir, negative_image_dir, part_image_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # 样本描述存储路径
    positive_anno_filename = os.path.join(save_path, str(face_size), "positive.txt")
    negative_anno_filename = os.path.join(save_path, str(face_size), "negative.txt")
    part_anno_filename = os.path.join(save_path, str(face_size), "part.txt")

    positive_count = 0
    negative_count = 0
    part_count = 0

    try:
        positive_anno_file = open(positive_anno_filename, "w")
        negative_anno_file = open(negative_anno_filename, "w")
        part_anno_file = open(part_anno_filename, "w")

        for i, line in enumerate(open(anno_src)):
            if i < 2:
                continue
            try:
                strs = line.strip().split()
                image_filename = strs[0].strip()
                print(image_filename)
                image_file = os.path.join(img_dir, image_filename)
                img = Image.open(image_file)
                img_w, img_h = img.size
                x1 = float(strs[1].strip())
                y1 = float(strs[2].strip())
                w = float(strs[3].strip())
                h = float(strs[4].strip())
                x2 = float(x1 + w)
                y2 = float(y1 + h)

                if max(w, h) < 40 or x1 < 0 or y1 < 0 or w < 0 or h < 0:
                    continue
                if w/h < 0.64 or w/h > 1.6:
                    continue
                boxes = np.array([[x1, y1, x2, y2]])
                # 计算出人脸中心点位置
                cx = x1 + w / 2
                cy = y1 + h / 2
                n_positive = n_part = n_negative = 0
                while True:
                    # 让人脸中心点有少许的偏移
                    w_ = np.random.randint(-w * 0.25, w * 0.25)
                    h_ = np.random.randint(-h * 0.25, h * 0.25)
                    cx_ = cx + w_
                    cy_ = cy + h_

                    # 让人脸形成正方形，并且让坐标也有少许的偏离
                    side_len = np.random.randint(int(min(w, h) * 0.6), np.ceil(1.25 * max(w, h)))
                    x1_ = np.max(cx_ - side_len / 2, 0)
                    y1_ = np.max(cy_ - side_len / 2, 0)
                    x2_ = x1_ + side_len
                    y2_ = y1_ + side_len

                    crop_box = np.array([x1_, y1_, x2_, y2_])

                    # 计算坐标的偏移值
                    offset_x1 = (x1 - x1_) / side_len
                    offset_y1 = (y1 - y1_) / side_len
                    offset_x2 = (x2 - x2_) / side_len
                    offset_y2 = (y2 - y2_) / side_len
                    # 剪切下图片，并进行大小缩放
                    face_crop = img.crop(crop_box)
                    face_resize = face_crop.resize((face_size, face_size))

                    iou = utils.iou(crop_box, boxes)[0]
                    if iou > 0.6 and n_positive < 5:  # 正样本
                        positive_anno_file.write(
                            "positive/{0}.jpg {1} {2} {3} {4} {5}\n".format(
                                positive_count, 1, offset_x1, offset_y1,
                                offset_x2, offset_y2))
                        positive_anno_file.flush()
                        face_resize.save(os.path.join(positive_image_dir, "{0}.jpg".format(positive_count)))
                        positive_count += 1
                        n_positive += 1
                    elif (iou > 0.25 and iou < 0.45) and (n_part < 5):  # 部分样本
                        part_anno_file.write(
                            "part/{0}.jpg {1} {2} {3} {4} {5}\n".format(
                                part_count, 2, offset_x1, offset_y1, offset_x2,
                                offset_y2))
                        part_anno_file.flush()
                        face_resize.save(os.path.join(part_image_dir, "{0}.jpg".format(part_count)))
                        part_count += 1
                        n_part += 1
                    if n_negative < 5:
                        box_area = w*h
                        img_area = img_w*img_h
                        side_len = np.random.randint(15, min(img_w, img_h) / 2)
                        x_ = np.random.randint(0, img_w - side_len)
                        y_ = np.random.randint(0, img_h - side_len)
                        if (box_area/img_area > 0.75) or (img_w/img_h>0.85 and img_w/img_h<1.15):
                            crop_box = np.array([x_, y_, x_+15, y_+15])
                        else:
                            crop_box = np.array([x_, y_, x_ + side_len, y_ + side_len])
                        if np.max(utils.iou(crop_box, boxes, isMin=True) < 0.2) or side_len < 20:
                            face_crop = img.crop(crop_box)
                            face_resize = face_crop.resize((face_size, face_size), Image.ANTIALIAS)

                            negative_anno_file.write("negative/{0}.jpg {1} 0 0 0 0\n".format(negative_count, 0))
                            negative_anno_file.flush()
                            face_resize.save(os.path.join(negative_image_dir, "{0}.jpg".format(negative_count)))
                            negative_count += 1
                            n_negative += 1

                    if n_positive + n_part + n_negative == 15:
                        break
            except Exception as e:
                traceback.print_exc()

    finally:
        positive_anno_file.close()
        negative_anno_file.close()
        part_anno_file.close()

