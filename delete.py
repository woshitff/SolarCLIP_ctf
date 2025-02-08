import os
import shutil

# 目标目录（当前目录）
target_dir = "."

# 遍历目标目录中的所有文件和文件夹
for item in os.listdir(target_dir):
    item_path = os.path.join(target_dir, item)
    
    # 判断是否是以 "pymp" 开头的文件夹
    if os.path.isdir(item_path) and item.startswith("pymp-"):
        # print(f"Deleting {item_path}")
        try:
            shutil.rmtree(item_path)  # 递归删除整个文件夹
            print(f"Deleted {item_path}")
        except Exception as e:
            print(f"Failed to delete {item_path}: {e}")
            pass  # 直接跳过
