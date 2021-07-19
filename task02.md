## 主要的提高分数/识别率的方式有以下几种：
1。调整模型参数并重新finetune
2。改进检测流程（如前置方向检测等）
3。在其他模型的基础上进行finetune
4。添加均衡化和透视画变化等预处理
5。自行搭建更好的模型结构，并从零开始训练


## 常见的图像增强方式
1。padding+crop
对图像做padding，再随机crop，可以减少检测模型在检测过程中产生对检测结果不稳定，文字目标在整图位置中的偏移带来的影响；
padding就是先对图像进行外延填充，如top padding、left padding以及same padding（全部padding）。就是在图像块的周围加上格子，使得图像的边缘数据也能被利用到。
例如：
图片: https://uploader.shimo.im/f/flN7hIjTZNqE9xwz.png?accessToken=eyJhbGciOiJIUzI1NiIsImtpZCI6ImRlZmF1bHQiLCJ0eXAiOiJKV1QifQ.eyJhdWQiOiJhY2Nlc3NfcmVzb3VyY2UiLCJleHAiOjE2MjY2MjQxMzUsImciOiJUZzlreThyZ0tKdzNKajY5IiwiaWF0IjoxNjI2NjIzODM1LCJ1c2VySWQiOjg3MTg4MTh9.NHw25T0yl7wVOIQ--EN_ec0DTzKzD1HDl9b9Krh5pXY
2。图像亮度对比度变化
通过提高或降低图像的亮度/对比度，减少识别效果受光线的影响，使得文在在图片上更清晰。
图片: https://uploader.shimo.im/f/GgWyt3LpITPMZQQR.png?accessToken=eyJhbGciOiJIUzI1NiIsImtpZCI6ImRlZmF1bHQiLCJ0eXAiOiJKV1QifQ.eyJhdWQiOiJhY2Nlc3NfcmVzb3VyY2UiLCJleHAiOjE2MjY2MjQxMzUsImciOiJUZzlreThyZ0tKdzNKajY5IiwiaWF0IjoxNjI2NjIzODM1LCJ1c2VySWQiOjg3MTg4MTh9.NHw25T0yl7wVOIQ--EN_ec0DTzKzD1HDl9b9Krh5pXY
详细代码如下：
def contrast_brightness(self, im: np.ndarray, c, b) -> tuple:#c是对比度，b是亮度， b值越大越亮
    """
    对彩色图片进行颜色变换
    :param im: 原始灰度图
    :return: 颜色进行变换后的彩色图片
    """
    h, w, c = im.shape
    blank = np.zeros([h, w, c], im.dtype)
    dst = cv2.addWeighted(im, c, blank, 1- c, b)
    return dst

3。图像直方图均衡化
是一种简单有效的图像增强技术，通过改变图像的直方图来改变图像中各个像素的灰度，主要是用来增强动态范围偏小的图像的对比度。原始图像可能由于其灰度分布集中在较窄的区间，造成图像不够清晰。（该技术也可用于彩色图片上）
图片: https://uploader.shimo.im/f/clYWVK9312azMcuI.png?accessToken=eyJhbGciOiJIUzI1NiIsImtpZCI6ImRlZmF1bHQiLCJ0eXAiOiJKV1QifQ.eyJhdWQiOiJhY2Nlc3NfcmVzb3VyY2UiLCJleHAiOjE2MjY2MjQxMzUsImciOiJUZzlreThyZ0tKdzNKajY5IiwiaWF0IjoxNjI2NjIzODM1LCJ1c2VySWQiOjg3MTg4MTh9.NHw25T0yl7wVOIQ--EN_ec0DTzKzD1HDl9b9Krh5pXY
详细代码如下：
    def histogram_equalization(self, im: np.ndarray):
        """
        rgb直方图均衡化
        :param im: 输入彩色图片
        :return: 返回均衡化的图片
        """
        # 彩色图像均衡化,需要分解通道 对每一个通道均衡化
        (b, g, r) = cv2.split(im)
        bH = cv2.equalizeHist(b)
        gH = cv2.equalizeHist(g)
        rH = cv2.equalizeHist(r)
        # 合并每一个通道
        result = cv2.merge((bH, gH, rH))
        return result

