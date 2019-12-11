import cv2
from PIL import Image

name = 'name'


def getface(path):
    cap = cv2.VideoCapture(path)
    classfier = cv2.CascadeClassifier("")
    suc = cap.isOpened()
    frame_count = 0
    out_count = 0
    while suc:
        frame_count += 1
        if out_count > 599:
            break
        suc, frame = cap.read()
        params = []
        params.append(2)
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faceRects = classfier.detectMultiScale(grey, 1.2, 3, (32, 32))
        if len(faceRects) > 0:
            for faceRect in faceRects:
                x, y, w, h = faceRect
                image = frame[y - 10:y + h + 10, x - 10, x + w + 10]
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                img_new = cv2.resize(image,(47,47),cv2.INTER_CUBIC)
                cv2.imwrite('/home/coding/workspace/Data'+name+'%d.jpg' % out_count,img_new,params)
                out_count+=1
                print('成功提取'+name+"的第%d个脸部"%out_count)
                break
    cap.release()
    cv2.destroyAllWindows()
    print('总帧数',frame_count)
    print('提取脸部',out_count)

if __name__ == '__main__':
    getface()
