from PIL import Image
import os
# 定义字符集
charSet = '''@#$%&?*aeoc=<{[(/l|!-_:;,."'^~` '''
# charSet = '''$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. '''


# 把灰度值转成字符
def Gary2Char(gray):
    if gray > 240 :
        return ' '
    else:
        return charSet[int(gray/((256.0+1)/len(charSet)))]
# 将图片转为字符图片
# 输入为文件路径，输出为字符图片的txt文件
def Image2Char(imagePath):
    # 读取图片
    image = Image.open(imagePath)
    # 改变图片大小
    image = image.resize((80,50),Image.ANTIALIAS)
    # 转为灰色图像
    imageL = image.convert('L')
    # print(imageL)
    # 保存字符图片
    imageChar = ''
    for i in range(imageL.height):
        for j in range(imageL.width):
            gray = imageL.getpixel((j,i))
            if isinstance(gray,tuple):
                gray = int(0.2126 * gray[0]+0.7125 * gray[1] + 0.0722*gray[2])
            imageChar+=Gary2Char(gray)
        imageChar+='\n'
    return imageChar
if __name__ == '__main__':
    # 获取当前文件地址
    cwd = os.getcwd()
    imageName = input('请输入文件名称：\n')
    # 图片地址
    # imagePath= 'F:\\newcoder\\Python\\Practice\\02.jpg'
    imagePath = cwd + '\\' + imageName + '.jpg'
    # 字符图片保存地址
    imageCharPath = imagePath[:-4] + '.txt'
    with open(imageCharPath,'w') as file :
        imageChar = Image2Char(imagePath)
        file.write(imageChar)