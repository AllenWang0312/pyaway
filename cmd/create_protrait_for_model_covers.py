import cv2
import os
import time
import shutil
import conf

# print(sys.argv[0])
def getAllPath(dirpath, *suffix):
    PathArray = []
    FileNames = []
    for r, ds, fs in os.walk(dirpath):
        for fn in fs:
            if os.path.splitext(fn)[1] in suffix:
                print(r,fn)
                fname = r+"/"+fn
                PathArray.append(fname)
                FileNames.append(fn)
    return PathArray,FileNames


def readPicSaveFace(sourcePath, targetPath, invalidPath, *suffix):
    imagePaths,fileNames = getAllPath(sourcePath, *suffix)
    print(imagePaths)
    try:
        count = 1
        face_cascade = cv2.CascadeClassifier()
        # a =face_cascade.load('F:\github\opencv\data\haarcascades\haarcascade_frontalface_alt.xml')
        a =face_cascade.load('E:\github\opencv-master\data\haarcascades\haarcascade_frontalface_alt.xml')
        print(a)
        for imagePath in imagePaths:
            img = cv2.imread(imagePath)
            if type(img) != str:
                faces = face_cascade.detectMultiScale(img, 1.1, 5)
                if len(faces):
                    for (x, y, w, h) in faces:
                        if w >= 64 and h >= 64:
                            fileName =os.path.splitext(fileNames[imagePaths.index(imagePath)])[0]
                            X = int(x * 0.5)
                            W = min(int((x + w) * 1.2), img.shape[1])
                            Y = int(y * 0.3)
                            H = min(int((y + h) * 1.4), img.shape[0])
                            f = cv2.resize(img[Y:H, X:W], (W - X, H - Y))
                            cv2.imwrite(targetPath + os.sep + '%s.jpg' % fileName, f)
                            count += 1
                            print(imagePath + "have face")
                else:
                    print(imagePath + "not found face")
                    shutil.move(imagePath, invalidPath)
    except IOError:
        print("Error")

    else:
        print('Find' + str(count - 1) + ' faces to Destination' + targetPath)


if __name__ == '__main__':
    invalidPath = conf.muri_path+'findno'
    readPicSaveFace(conf.cover_path, conf.save_path, invalidPath, '.jpg', '.JPG', 'png', 'PNG')
