# 导入所需模块
import cv2
from matplotlib import pyplot as plt
import os
import numpy as np


# 蓝色阈值
def color_position_blue(img):
    colors = [
              ([100, 110, 110], [130, 255, 255]),  # 蓝色
              ]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for (lower, upper) in colors:
        lower = np.array(lower, dtype="uint8")  # 颜色下限
        upper = np.array(upper, dtype="uint8")  # 颜色上限

        # 根据阈值找到对应的颜色
        mask = cv2.inRange(hsv, lowerb=lower, upperb=upper)
        output = cv2.bitwise_and(img, img, mask=mask)
    return output

#绿色阈值
def color_position_green(img):
    colors = [
              ([60, 100, 70], [150, 200, 180]),  # 绿色
              ]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for (lower, upper) in colors:
        lower = np.array(lower, dtype="uint8")  # 颜色下限
        upper = np.array(upper, dtype="uint8")  # 颜色上限

        # 根据阈值找到对应的颜色
        mask = cv2.inRange(hsv, lowerb=lower, upperb=upper)
        output = cv2.bitwise_and(img, img, mask=mask)
    return output


# 提取车牌部分图片
def get_carLicense_img_green(image):
    # 高斯模糊处理
    image = color_position_green(image)
    image = cv2.GaussianBlur(image, (3,3), 0)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # sobel算子边缘检测（做了一个y方向的检测）
    Sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0)
    Sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1)
    absX = cv2.convertScaleAbs(Sobel_x)
    absY = cv2.convertScaleAbs(Sobel_y)
    image = absY
    # plt.subplot(2, 2, 2)
    # plt.title('gray')
    # plt.imshow(image)

    # 自适应阈值处理
    ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)

    # 闭运算,是白色部分练成整体
    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernelX,iterations = 3)
    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 19))
    # 膨胀腐蚀操作
    #image = cv2.erode(image, kernelY)
    image = cv2.dilate(image, kernelY)
    image = cv2.dilate(image, kernelY)
    image = cv2.erode(image, kernelY)
    # plt.subplot(2, 2, 3)
    # plt.title('gray')
    # plt.imshow(image,cmap='gray')
    # 中值滤波去除噪点
    image = cv2.medianBlur(image, 15)
    # 轮廓检测
    # cv2.RETR_EXTERNAL表示只检测外轮廓
    # cv2.CHAIN_APPROX_SIMPLE压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 绘制轮廓
    for item in contours:
        rect = cv2.boundingRect(item)
        x = rect[0]
        y = rect[1]
        weight = rect[2]
        height = rect[3]
        if (weight > (height * 2)) and (weight < (height * 5.5)):
            image = origin_image[y-int(0.4*height):y + height, x:x + weight]
            return image
    return image

def get_carLicense_img_blue(image):
    # 高斯模糊处理
    image = color_position_blue(image)
    image = cv2.GaussianBlur(image, (3,3), 0)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # sobel算子边缘检测（做了一个y方向的检测）
    Sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0)
    Sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1)
    absX = cv2.convertScaleAbs(Sobel_x)
    absY = cv2.convertScaleAbs(Sobel_y)
    image = absY
    # plt.subplot(2, 2, 2)
    # plt.title('gray')
    # plt.imshow(image)

    # 自适应阈值处理
    ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)

    # 闭运算,是白色部分练成整体
    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernelX,iterations = 3)
    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 19))
    # 膨胀腐蚀操作
    #image = cv2.erode(image, kernelY)
    image = cv2.dilate(image, kernelY)
    image = cv2.dilate(image, kernelY)
    image = cv2.erode(image, kernelY)
    # plt.subplot(2, 2, 3)
    # plt.title('gray')
    # plt.imshow(image,cmap='gray')
    # 中值滤波去除噪点
    image = cv2.medianBlur(image, 15)
    # 轮廓检测
    # cv2.RETR_EXTERNAL表示只检测外轮廓
    # cv2.CHAIN_APPROX_SIMPLE压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 绘制轮廓
    for item in contours:
        rect = cv2.boundingRect(item)
        x = rect[0]
        y = rect[1]
        weight = rect[2]
        height = rect[3]
        if (weight > (height * 2)) and (weight < (height * 5.5)):
            image = origin_image[y:y + height, x:x + weight]
            return image
    return False
# 读取待检测图片
origin_image = cv2.imread('../img/6.jpg')
b, g, r = cv2.split(origin_image)
img = cv2.merge([r, g, b])
plt.subplot(2,1,1)
plt.title('origin')
plt.imshow(img)
image = origin_image.copy()
carLicense_image = get_carLicense_img_blue(image)
if type(carLicense_image) != np.ndarray:
    carLicense_image = get_carLicense_img_green(image)
plt.subplot(2,1,2)
plt.title('chepai')
plt.imshow(carLicense_image[...,::-1])
cv2.imwrite('../img/chepairesult.jpg',carLicense_image)
plt.show()


from aip import AipOcr
APP_ID = '20742056'
API_KEY = 'EnlGTLZvivZAl9qNoeKbf7sU'
SECRET_KEY = 'PkNL8zfb6WqdIlNIm3A5HZaGpShhWuzC'
client = AipOcr(APP_ID, API_KEY, SECRET_KEY)


def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()


def image2text(fileName):
    image = get_file_content(fileName)
    dic_result = client.basicGeneral(image)
    res = dic_result['words_result']
    result = ''
    for m in res:
        result = result + str(m['words'])
    return result


getresult = image2text('../img/chepairesult.jpg')
print(getresult)
