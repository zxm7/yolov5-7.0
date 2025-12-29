import os
import shutil

# 1. 路径定义
base_path = '/home/zhangxu/yolov5-7.0/datasets/UTDAC2020'
# 我们只扫描当前的 UTDAC2020 目录，绝不乱跑
search_dirs = ['images/train', 'images/val', 'images/test']
pool_img = '/home/zhangxu/yolov5-7.0/datasets/CLEAN_POOL/images'
pool_lab = '/home/zhangxu/yolov5-7.0/datasets/CLEAN_POOL/labels'


def recover_data():
    unique_names = set()
    print("正在从乱序的文件夹中提取唯一数据源...")

    for sub in search_dirs:
        img_src_dir = os.path.join(base_path, sub)
        lab_src_dir = img_src_dir.replace('images', 'labels')

        if not os.path.exists(img_src_dir): continue

        for f in os.listdir(img_src_dir):
            if f.lower().endswith('.jpg'):
                name = f.rsplit('.', 1)[0]
                if name not in unique_names:
                    # 只有第一次发现该文件名时，才搬运
                    unique_names.add(name)
                    # 搬运图片
                    shutil.move(os.path.join(img_src_dir, f), os.path.join(pool_img, f))
                    # 搬运对应的标签
                    old_lab = os.path.join(lab_dir_search(sub, name))
                    if old_lab and os.path.exists(old_lab):
                        shutil.move(old_lab, os.path.join(pool_lab, name + '.txt'))

    print(f"✅ 成功提取出 {len(unique_names)} 组唯一数据到 CLEAN_POOL！")
    print("正在物理清空原有的所有 train/val/test 冗余文件...")

    # 彻底删掉原有的 images/train, images/val 等，确保空空如也
    for d in ['images', 'labels']:
        for s in ['train', 'val', 'test']:
            target = os.path.join(base_path, d, s)
            shutil.rmtree(target, ignore_errors=True)
            os.makedirs(target, exist_ok=True)


def lab_dir_search(sub, name):
    # 辅助寻找标签的逻辑
    p = os.path.join(base_path, sub.replace('images', 'labels'), name + '.txt')
    return p if os.path.exists(p) else None


if __name__ == '__main__':
    recover_data()