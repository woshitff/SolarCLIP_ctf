#!/bin/bash

# 定义文件列表
# files=("0094" "0131" "0171" "0193" "0211" "0304" "0335" "1600" "1700" "4500" "hmi")
folders=("0094" "0131" "0171" "0193" "0211" "0304" "0335" "1600" "1700" "4500" "hmi")

# 定义源路径和目标路径
# source_base_path="/mnt/tianwen-tianqing-nas/tianwen/ctf/solarclip/zyz_106/logs/reconmodels/autoencoder/vae"
target_base_path="/mnt/tianwen-tianqing-nas/tianwen/ctf/solarclip/ctf_105/SolarCLIP_ctf/checkpoints/reconmodels/ae/two_stage"

# # 循环遍历每个文件并执行复制操作
# for file in "${files[@]}"
# do
#     source_file="${source_base_path}/ae${file}_second/checkpoints/trainstep_checkpoints/best_val_loss_epoch.ckpt"  # 源文件路径
#     target_file="${target_base_path}/${file}/best_val_loss_epoch.ckpt"  # 目标文件路径
    
#     # 执行复制操作
#     echo "Copying from ${source_file} to ${target_file}"
#     sudo cp "$source_file" "$target_file"
    
#     # 检查复制是否成功
#     if [ $? -eq 0 ]; then
#         echo "Successfully copied ${source_file} to ${target_file}"
#     else
#         echo "Failed to copy ${source_file} to ${target_file}"
#     fi
# done
# for folder in "${folders[@]}"
# do 
#     dir_path="${target_base_path}/${folder}"
#     mkdir -p "$dir_path"
# done
for folder in "${folders[@]}"
do
    # 定义文件路径
    file_path="${target_base_path}/${folder}/best_val_loss_epoch.ckpt"

    # 检查文件是否存在
    if [ -f "$file_path" ]; then
        # 为文件添加可读权限
        echo "Adding read permission for $file_path"
        sudo chmod +r "$file_path"
        
        # 检查权限是否成功
        if [ $? -eq 0 ]; then
            echo "Successfully added read permission to $file_path"
        else
            echo "Failed to add read permission to $file_path"
        fi
    else
        echo "File $file_path does not exist!"
    fi
done