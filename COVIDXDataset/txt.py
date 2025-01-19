import os

def filter_text_by_image(input_txt_path, images_folder, output_txt_path):
    """
    过滤文本文件中包含的图片信息，根据图片是否存在于指定文件夹中。

    :param input_txt_path: 输入文本文件路径
    :param images_folder: 图片文件夹路径
    :param output_txt_path: 输出文本文件路径
    """
    # 确保文件路径和文件夹路径存在
    if not os.path.exists(input_txt_path):
        print(f"输入文件不存在: {input_txt_path}")
        return

    if not os.path.isdir(images_folder):
        print(f"指定的图片文件夹不存在: {images_folder}")
        return

    # 创建输出文件
    with open(input_txt_path, 'r', encoding='utf-8') as infile, \
         open(output_txt_path, 'w', encoding='utf-8') as outfile:

        for line in infile:
            line = line.strip()  # 去除换行符和多余空格
            if not line:
                continue  # 跳过空行

            parts = line.split()  # 按空格分割
            if len(parts) < 2:
                continue  # 跳过无效行

            image_name = parts[1]  # 第二部分为图片名称
            image_path = os.path.join(images_folder, image_name)

            if os.path.exists(image_path):  # 如果图片存在，写入输出文件
                outfile.write(line + '\n')

    print(f"过滤完成，结果保存在: {output_txt_path}")

# 示例调用
input_txt = r"D:\UPM\bayesiannet\COVID_BayesianNET\data\test_split1.txt"  # 输入文件路径
images_dir = r"D:\UPM\bayesiannet\COVID_BayesianNET\data\test"  # 图片文件夹路径
output_txt = r"D:\UPM\bayesiannet\COVID_BayesianNET\data\test_split2.txt"  # 输出文件路径

filter_text_by_image(input_txt, images_dir, output_txt)
