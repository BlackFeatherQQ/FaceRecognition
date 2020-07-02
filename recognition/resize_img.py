# import numpy as np
# a = np.arange(12).reshape([2,2,3])
# print(a)
# print(a[:,:,0])
# print(a[...,0])

from PIL import Image
import os



def pad_image(image, target_size):
    iw, ih = image.size  # 原始图像的尺寸
    w, h = target_size  # 目标图像的尺寸
    scale = min(float(w) / float(iw), float(h) / float(ih))  # 转换的最小比例

    # 保证长或宽，至少一个符合目标图像的尺寸
    nw = int(iw * scale)
    nh = int(ih * scale)

    # image = image.resize((nw, nh), Image.BICUBIC)  # 采用双三次插值算法缩小图像
    image = image.resize(target_size, Image.BICUBIC)  # 采用双三次插值算法缩小图像
    # image.show()
    # new_image = Image.new('RGB', target_size, (128, 128, 128))  # 生成灰色图像
    # new_image.show()
    # // 为整数除法，计算图像的位置
    # new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))  # 将图像填充为中间图像，两侧为灰色的样式
    # new_image.paste(image, (0, 0))  # 将图像填充为(0,0)点，两侧为灰色的样式
    # new_image.show()

    return image.convert("RGB"),scale


def main(path,save_path):
    index = 0
    for i in os.listdir(path):

        img_path = os.path.join(path, i)
        new_img_path = os.path.join(save_path, f"{index}.PNG")
        image = Image.open(img_path)
        size = (160, 160)
        tt,_ = pad_image(image, size)  # 填充图像
        # tt.show()
        tt.save(new_img_path)
        new_img_path = os.path.join(save_path, f"{index+1}.PNG")
        image =  image.transpose(Image.FLIP_LEFT_RIGHT)
        tt, _ = pad_image(image, size)  # 填充图像
        tt.save(new_img_path)
        index = index+2


if __name__ == '__main__':
    main("D:/code_data/face_recognition/orgin_img/hu_ge_png","D:/code_data/face_recognition/images/hu_ge")
