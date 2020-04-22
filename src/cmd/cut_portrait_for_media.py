import cv2
import os
from PIL import Image

from configparser import ConfigParser
# print(sys.argv[0])

proDir = os.path.split(os.path.realpath(__file__))[0]
# proDir = os.path.dirname(os.path.realpath(__file__))  与上面一行代码作用一样
configPath = os.path.join(proDir, "../../conf/config.ini")
path = os.path.abspath(configPath)
cp = ConfigParser()
cp.read(path)
disk_path = cp.get("workspace", "disk_path")
opencv_path = cp.get("workspace", "opencv_path")

name = 'name'

def getface(path):
    cap = cv2.VideoCapture(path)
    classfier = cv2.CascadeClassifier()
    b = classfier.load(opencv_path+"\opencv\data\haarcascades\haarcascade_frontalface_alt2.xml")
    print(b)
    suc = cap.isOpened()
    frame_count = 0
    out_count = 0
    while suc:
        frame_count += 1
        if out_count > 599:
            break
        suc, frame = cap.read()
        params = []
        # params.append(2)
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faceRects = classfier.detectMultiScale(grey, 1.2, 3,0,(300, 300))
        if len(faceRects) > 0:
            for faceRect in faceRects:
                x, y, w, h = faceRect
                image = frame[y - 10:y + h + 10, x - 10: x + w + 10]
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                img_new = cv2.resize(image,(100,100),cv2.INTER_CUBIC)
                cv2.imwrite(disk_path+'/aboutu/records/'+name+'%d.jpg' % out_count,img_new,params)
                out_count+=1
                print('成功提取'+name+"的第%d个脸部"%out_count)
                break
    cap.release()
    cv2.destroyAllWindows()
    print('总帧数',frame_count)
    print('提取脸部',out_count)

if __name__ == '__main__':
    getface(disk_path+'/aboutu/records/20181028-213112.mp4')
