import numpy as np
import cv2
import matplotlib.pyplot as plt

base_dir = 'D:\\pythonStudy\\orl_face'
save_dir = 'D:\\pythonStudy\\secretPic'


# 制作加密图像数据库
def setDCTPictures():
    for i in range(1, 41):
        j = 1
        while j <= 10:
            normalPic_dir = base_dir + '\\s' + str(i) + '\\' + str(j) + '.pgm'
            y = cv2.imread(normalPic_dir, 0)
            y1 = np.float32(y)
            Y = cv2.dct(y1)
            # 进行logistic混沌->再将图片存起来
            L = logisticPic(Y)
            # 对系数矩阵Ｌ（ｉ，ｊ）做ＩＤＣＴ变换即可得到加密后的人脸图像Ｅ（ｉ，ｊ）
            L = cv2.idct(L)
            L = np.asarray(L).astype(np.uint8)
            # 解密way：同样的系数对原来的图片再加密一次
            # N = cv2.dct(L)
            # N = logisticPic(N)
            # N = cv2.idct(N)
            # N = np.asarray(N).astype(np.uint8)
            cv2.imwrite(save_dir + '\\' + str(i) + '_' + str(j) + '.png', L)
            getPromotionmatrix(L)
            j = j + 1
        i = i + 1



# Logistic
def logisticPic(pic):
    # 产生混沌序列 迭代300次（前文）
    u = 4.0
    x0 = 0.135
    xi = [x0]
    for i in range(1, 10304):
        xk1 = u * xi[i - 1] * (1 - xi[i - 1])
        xi.append(xk1)
    # 对混沌序列Ｘ(i)进行二值化，根据图像大小构造二值加密矩阵sign(x)
    sign = []
    for i in range(len(xi)):
        if xi[i] >= 0.5:
            temp = 1
        else:
            temp = -1
        sign.append(temp)
    sign = np.array(sign).reshape(112, 92)
    # 频域系数矩阵Ｆ（ｉ，ｊ）与二值加密矩阵ｓｉｇｎ（ｘ）做点乘得到系数矩阵Ｌ（ｉｊ）
    L = pic * sign
    return L


# 对加密后的训练样本图像利用PCA算法提取特征，得到投影矩阵Ｔ
def getPromotionmatrix(X):
    # 获取维数
    num_data, dim = X.shape
    # 数据中心化
    mean_X = X.mean(axis=0)
    X = X - mean_X
    if dim > num_data:  # PCA- 使用紧致技巧
        M = np.dot(X, X.T)  # 协方差矩阵
        e, EV = np.linalg.eigh(M)  # 特征值和特征向量
        tmp = np.dot(X.T, EV).T  # 这就是紧致技巧
        V = tmp[::-1]  # 由于最后的特征向量是我们所需要的，所以需要将其逆转
        S = np.sqrt(e)[::-1]  # 由于特征值是按照递增顺序排列的，所以需要将其逆转
        for i in range(V.shape[1]):
            V[:, i] /= S
    else:  # PCA- 使用SVD 方法
        U, S, V = np.linalg.svd(X)
        V = V[:num_data]  # 仅仅返回前nun_data 维的数据才合理

    # 返回投影矩阵
    return V


# 对划分好的训练样本Ｘ与投影矩阵相乘，得到训练样本的降维矩阵Ｄ
def getDimensionReductionMatrix():
    return


# 将降维矩阵Ｄ作为神经网络的输入，创建并训练网络
def trainNetWork():
    return


def main():
    setDCTPictures()


if __name__ == "__main__":
    main()
