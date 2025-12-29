import os
import random
import shutil

pool_path = '/home/zhangxu/yolov5-7.0/datasets/CLEAN_POOL'
target_base = '/home/zhangxu/yolov5-7.0/datasets/UTDAC2020'

def final_split():
    imgs = [f for f in os.listdir(os.path.join(pool_path, 'images')) if f.endswith('.jpg')]
    random.seed(42)
    random.shuffle(imgs)

    total = len(imgs)
    t_end = int(total * 0.8)
    v_end = int(total * 0.9)

    tasks = {
        'train': imgs[:t_end],
        'val': imgs[t_end:v_end],
        'test': imgs[v_end:]
    }

    for sub, files in tasks.items():
        for f in files:
            name = f.rsplit('.', 1)[0]
            # 移回 images
            shutil.move(os.path.join(pool_path, 'images', f),
                        os.path.join(target_base, 'images', sub, f))
            # 移回 labels
            lab_f = name + '.txt'
            src_l = os.path.join(pool_path, 'labels', lab_f)
            if os.path.exists(src_l):
                shutil.move(src_l, os.path.join(target_base, 'labels', sub, lab_f))
        print(f"已完成 {sub} 集分发: {len(files)} 张")

    # 清理掉已经空了的回收池
    shutil.rmtree(pool_path)
    print("✨ 所有错误已消除，数据已还原为精准的 8:1:1 结构！")

if __name__ == '__main__':
    final_split()