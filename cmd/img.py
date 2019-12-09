import cv2
import os
import time
import shutil


def getAllPath(dirpath, *suffix):
    PathArray = []
    for r, ds, fs in os.walk(dirpath):
        for fn in fs:
            if os.path.splitext(fn)[1] in suffix:
                fname = os.path.join(r, fn)
                print(fname)
                PathArray.append(fname)
    return PathArray


def readPicSaveFace(sourcePath, targetPath, invalidPath, *suffix):
    imagePaths = getAllPath(sourcePath, *suffix)
    try:
        count = 1
        face_cascade = cv2.CascadeClassifier('F:\\github\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt.xml')
        for imagePath in imagePaths:
            img = cv2.imread(imagePath)
            if type(img) != str:
                faces = face_cascade.detectMultiScale(img, 1.1, 5)
                if len(faces):
                    for (x, y, w, h) in faces:
                        if w >= 128 and h >= 128:
                            listStr = [str(int(time.time())), str(count)]
                            fileName = ''.join(listStr)
                            X = int(x * 0.5)
                            W = min(int((x + w) * 1.2), img.shape[1])
                            Y = int(y * 0.3)
                            H = min(int((y + h) * 1.4), img.shape[0])
                            f = cv2.resize(img[Y:H, X:W], (W - X, H - Y))
                            cv2.imwrite(targetPath + os.sep + '%s.jpg' % fileName, f)
                            count += 1
                            print(imagePath + "have face")
                else:
                    shutil.move(imagePath, invalidPath)
    except IOError:
        print("Error")

    else:
        print('Find' + str(count - 1) + ' faces to Destination' + targetPath)


if __name__ == '__main__':
    invalidPath = 'F:\\work\\muri\\haveNoPeaple'
    sourcePath = 'F:\\work\\muri\\cover'
    targetPath = 'F:\\work\\muri\\faceOfPeaple'
    readPicSaveFace(sourcePath, targetPath, invalidPath, '.jpg', '.JPG', 'png', 'PNG')
