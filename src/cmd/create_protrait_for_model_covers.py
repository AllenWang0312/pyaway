import cv2
import os
import time
import shutil
from configparser import ConfigParser

# print(sys.argv[0])

proDir = os.path.split(os.path.realpath(__file__))[0]
# proDir = os.path.dirname(os.path.realpath(__file__))  与上面一行代码作用一样
print(proDir)
configPath = os.path.join(proDir, "../../conf/config.ini")
path = os.path.abspath(configPath)
print(configPath)
print(path)

cp = ConfigParser()
cp.read(path)

disk_path = cp.get("workspace", "disk_path")
opencv_path = cp.get("workspace", "opencv_path")

# muri_path = disk_path+"/aboutu/uu"
cover_path = disk_path + "/photos/meituri_cn/101"
save_path = disk_path + "/photos/meituri_cn/101_portrait"


# invalidPath = disk_path + '/photos/meituri_cn/0/19013_findno'


def getAllPath(dirpath, *suffix):
    PathArray = []
    FileNames = []
    for r, ds, fs in os.walk(dirpath):
        for fn in fs:
            print(r, fn)
            if os.path.splitext(fn)[1] in suffix:
                fname = r + "/" + fn
                PathArray.append(fname)
                FileNames.append(fn)
    return PathArray, FileNames


def readPicSaveFace(sourcePath, targetPath, *suffix):
    imagePaths, fileNames = getAllPath(sourcePath, *suffix)
    print(imagePaths)
    try:
        count = 1
        face_cascade = cv2.CascadeClassifier()
        # a =face_cascade.load('F:/github/opencv/data/haarcascades/haarcascade_frontalface_alt.xml')
        a = face_cascade.load(
            # opencv_path + '/opencv/data/haarcascades/haarcascade_frontalface_default.xml'//haarcascade_frontalface_alt2
            opencv_path + '/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml'
            # opencv_path + '/opencv/data/haarcascades/haarcascade_eye.xml'
        )
        print(a)
        for imagePath in imagePaths:
            img = cv2.imread(imagePath)
            if type(img) != str:
                faces = face_cascade.detectMultiScale(img, 1.5, 5, minSize=(64, 64), maxSize=(1024, 1024))
                if len(faces):
                    for (x, y, w, h) in faces:
                        if w >= 64 and h >= 64:
                            fileName = os.path.splitext(
                                fileNames[imagePaths.index(imagePath)])[0]
                            # X = int(x * 0.5)
                            # W = min(int((x + w) * 1.2), img.shape[1])
                            # Y = int(y * 0.3)
                            # H = min(int((y + h) * 1.4), img.shape[0])
                            # f = cv2.resize(img[Y:H, X:W], (W - X, H - Y))
                            f = cv2.resize(img[y:y+h, x:x+w], ( w, h))
                            cv2.imwrite(targetPath + os.sep +
                                        '%s.jpg' % count, f)
                            count += 1
                            print(imagePath + "have face")
                else:
                    print(imagePath + "not found face")
                    # shutil.move(imagePath, invalidPath)
    except IOError:
        print("Error")

    else:
        print('Find' + str(count - 1) + ' faces to Destination' + targetPath)


if __name__ == '__main__':
    # if os._exists(save_path):
    #     print("dir exist:"+save_path)
    # else:
    #     os.mkdir(save_path, 777)
    #     getAllPath(cover_path, '.jpg', '.JPG', '.png', '.PNG')
    readPicSaveFace(cover_path, save_path,
                    '.jpg', '.JPG', '.png', '.PNG')
