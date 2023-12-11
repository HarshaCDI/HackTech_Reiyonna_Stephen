import os
import shutil
import copy
import cv2
import xml.etree.ElementTree as ET
from tqdm import tqdm


def preprocess_data(src_root, dst_root):
    if not os.path.isdir(src_root):
        print("[Err]: invalid source root")
        return

    if not os.path.isdir(dst_root):
        os.makedirs(dst_root)
        print("{} made".format(dst_root))

    dst_img_dir_test = os.path.join(dst_root, 'images', 'test')

    if not os.path.isdir(dst_img_dir_test):
        os.makedirs(dst_img_dir_test)

    for seq_dir in os.listdir(src_root):
        seq_path = os.path.join(src_root, seq_dir)
        if os.path.isdir(seq_path):
            for img_file in os.listdir(seq_path):
                if img_file.endswith('.jpg'):
                    img_path = os.path.join(seq_path, img_file)
                    if os.path.isfile(img_path):
                        dst_seq_dir = os.path.join(dst_img_dir_test, seq_dir)
                        dst_img1_dir = os.path.join(dst_seq_dir, 'img1')

                        if not os.path.isdir(dst_img1_dir):
                            os.makedirs(dst_img1_dir)

                        dst_img_path = os.path.join(dst_img1_dir, img_file)

                        if os.path.isfile(dst_img_path):
                            print('{} already exists.'.format(dst_img_path))
                        else:
                            shutil.copy(img_path, dst_img1_dir)
                            print('{} copied to {}'.format(img_file, dst_img1_dir))


def blackout_ignore_regions(image, boxes):
    if image is None:
        print('[Err]: Input image is none!')
        return -1

    for box in boxes:
        box = list(map(lambda x: int(x + 0.5), box))
        image[box[1]: box[1] + box[3], box[0]: box[0] + box[2]] = [0, 0, 0]

    return image


def generate_labels(xml_root, img_root, label_root, viz_root=None):
    if not (os.path.isdir(xml_root) and os.path.isdir(img_root)):
        print('[Err]: invalid dirs')
        return -1

    xml_file_paths = [os.path.join(xml_root, x) for x in os.listdir(xml_root)]
    img_dirs = [os.path.join(img_root, x) for x in os.listdir(img_root)]
    xml_file_paths.sort()
    img_dirs.sort()
    print("Number of XML files:", len(xml_file_paths))
    print("Number of image directories:", len(img_dirs))

    assert (len(xml_file_paths) == len(img_dirs))

    track_start_id = 0
    frame_count = 0

    for xml_path, img_dir in zip(xml_file_paths, img_dirs):
        seq_max_target_id = 0
        if os.path.isfile(xml_path) and os.path.isdir(img_dir):
            if xml_path.endswith('.xml'):
                sub_dir_name = os.path.split(img_dir)[-1]
                if os.path.split(xml_path)[-1][:-4] != sub_dir_name:
                    print('[Err]: xml file and dir not match')
                    continue

                tree = ET.parse(xml_path)
                root = tree.getroot()
                seq_name = root.get('name')
                if seq_name != sub_dir_name:
                    print('[Warning]: xml file and dir not match')
                    continue
                print('Start processing seq {}...'.format(sub_dir_name))

                seq_label_root = os.path.join(label_root, seq_name, 'img1')
                if not os.path.isdir(seq_label_root):
                    os.makedirs(seq_label_root)
                else:
                    shutil.rmtree(seq_label_root)
                    os.makedirs(seq_label_root)

                seq_max_target_id = 0

                ignor_region = root.find('ignored_region')
                boxes = []

                for box_info in ignor_region.findall('box'):
                    box = [
                        float(box_info.get('left')),
                        float(box_info.get('top')),
                        float(box_info.get('width')),
                        float(box_info.get('height'))
                    ]
                    boxes.append(box)

                for frame in root.findall('frame'):
                    frame_count += 1
                    target_list = frame.find('target_list')
                    targets = target_list.findall('target')
                    density = int(frame.get('density'))

                    if density != len(targets):
                        print('[Err]: density not match @', frame)
                        return -1

                    img_id = int(frame.get('num'))

                    img_path = os.path.join(img_dir, 'img1', 'img{:05d}.jpg'.format(img_id))
                    if not os.path.isfile(img_path):
                        print('[Err]: image file not exists!')
                        return -1

                    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

                    if img is None:
                        print('[Err]: read image failed!')
                        return -1

                    img = blackout_ignore_regions(img, boxes)
                    cv2.imwrite(img_path, img)

                    if not (viz_root is None):
                        viz_path = os.path.join(viz_root, '{}_{}'.format(seq_name, os.path.split(img_path)[-1]))
                        img_viz = copy.deepcopy(img)

                    frame_label_strs = []

                    for target in targets:
                        target_id = int(target.get('id'))
                        if target_id > seq_max_target_id:
                            seq_max_target_id = target_id

                        track_id = target_id + track_start_id

                        bbox_info = target.find('box')
                        bbox_left = float(bbox_info.get('left'))
                        bbox_top = float(bbox_info.get('top'))
                        bbox_width = float(bbox_info.get('width'))
                        bbox_height = float(bbox_info.get('height'))

                        attr_info = target.find('attribute')
                        vehicle_type = str(attr_info.get('vehicle_type'))
                        trunc_ratio = float(attr_info.get('truncation_ratio'))

                        class_index = 0

                        if vehicle_type == 'car':
                            class_index = 0
                        elif vehicle_type == 'bus':
                            class_index = 1
                        elif vehicle_type == 'van':
                            class_index = 2
                        else:
                            class_index = 3

                        if not (viz_root is None):
                            pt_1 = (int(bbox_left + 0.5), int(bbox_top + 0.5))
                            pt_2 = (int(bbox_left + bbox_width), int(bbox_top + bbox_height))
                            cv2.rectangle(img_viz, pt_1, pt_2, (0, 255, 0), 2)

                        bbox_center_x = bbox_left + bbox_width * 0.5
                        bbox_center_y = bbox_top + bbox_height * 0.5

                        bbox_center_x /= img.shape[1]
                        bbox_center_y /= img.shape[0]
                        bbox_width /= img.shape[1]
                        bbox_height /= img.shape[0]

                        label_str = '{} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                            class_index, bbox_center_x, bbox_center_y, bbox_width, bbox_height)
                        frame_label_strs.append(label_str)

                    if not (viz_root is None):
                        cv2.imwrite(viz_path, img_viz)

                    label_f_path = os.path.join(seq_label_root, 'img{:05d}.txt'.format(img_id))
                    with open(label_f_path, 'w') as f:
                        for label_str in frame_label_strs:
                            f.write(label_str)

                print('Seq {} start track id: {:d}, has {:d} tracks'
                      .format(seq_name, track_start_id + 1, seq_max_target_id))
                track_start_id += seq_max_target_id
                print('Processing seq {} done.\n'.format(sub_dir_name))

    print('Total {:d} frames'.format(frame_count))


def generate_dot_test_file(data_root, rel_path, out_root):
    if not (os.path.isdir(data_root) and os.path.isdir(out_root)):
        print('[Err]: invalid root')
        return

    out_f_path = os.path.join(out_root, 'detrac.test')
    count = 0

    with open(out_f_path, 'w') as f:
        root = os.path.join(data_root, rel_path)
        seqs = [x for x in os.listdir(root)]
        seqs.sort()

        for seq in tqdm(seqs):
            img_dir = os.path.join(root, seq, 'img1')
            imgs = [x for x in os.listdir(img_dir)]
            imgs.sort()

            for img in imgs:
                if img.endswith('.jpg'):
                    img_path = os.path.join(img_dir, img)
                    if os.path.isfile(img_path):
                        item = img_path.replace(data_root + '/', '')
                        print(item)
                        f.write(item + '\n')
                        count += 1

    print('Total {:d} images for training'.format(count))


def find_files_with_suffix(root, suffix, file_list):
    for f in os.listdir(root):
        f_path = os.path.join(root, f)
        if os.path.isfile(f_path) and f.endswith(suffix):
            file_list.append(f_path)
        elif os.path.isdir(f_path):
            find_files_with_suffix(f_path, suffix, file_list)


def count_files_in_dirs(img_root, label_root):
    img_file_list, label_file_list = [], []

    find_files_with_suffix(img_root, '.jpg', img_file_list)
    find_files_with_suffix(label_root, '.txt', label_file_list)

    print('Total {:d} image files'.format(len(img_file_list)))
    print('Total {:d} label(txt) files'.format(len(label_file_list)))


def clean_test_set_mismatch(img_root, label_root):
    if not (os.path.isdir(img_root) and os.path.isdir(label_root)):
        print('[Err]: invalid root!')
        return

    img_dirs = [os.path.join(img_root, x) for x in os.listdir(img_root)]
    label_dirs = [os.path.join(label_root, x) for x in os.listdir(label_root)]

    assert (len(img_dirs) == len(label_dirs))

    img_dirs.sort()
    label_dirs.sort()

    for img_dir, label_dir in tqdm(zip(img_dirs, label_dirs)):
        for img_name in os.listdir(os.path.join(img_dir, 'img1')):
            txt_name = img_name.replace('.jpg', '.txt')
            txt_path = os.path.join(label_dir, 'img1', txt_name)
            img_path = os.path.join(img_dir, 'img1', img_name)

            if os.path.isfile(img_path) and os.path.isfile(txt_path):
                continue
            elif os.path.isfile(img_path) and (not os.path.isfile(txt_path)):
                os.remove(img_path)
                print('{} removed.'.format(img_path))
            elif os.path.isfile(txt_path) and (not os.path.isfile(img_path)):
                os.remove(txt_path)
                print('{} removed.'.format(txt_path))


if __name__ == '__main__':
    preprocess_data(
        src_root='../data/Insight-MVT_Annotation_Test',
        dst_root='../data/DETRAC'
    )

    generate_labels(
        xml_root='../data/DETRAC-Test-Annotations-XML',
        img_root='../data/DETRAC/images/test',
        label_root='../data/MOT/DETRAC/labels/test',
        viz_root='../UA_DETRAC_Hacktech_Final/data/viz_result'
    )

    clean_test_set_mismatch(
        img_root='../data/DETRAC/images/test',
        label_root='../data/DETRAC/labels/test'
    )

    generate_dot_test_file(
        data_root='../UA_DETRAC_Hacktech_Final/data/',
        rel_path='/DETRAC/images/test',
        out_root='../UA_DETRAC_Hacktech_Final/data'
    )

    count_files_in_dirs(
        img_root='../data/DETRAC/images/test',
        label_root='../data/DETRAC/labels'
    )

    print('Done')
