import cv2

testdir = '/media/wpc/H4_仓库/photos/meituri_cn/0/19013/5.jpg'

img = cv2.imread(testdir)

b , g , r = cv2.split(img)
merged = cv2.merge([b,g,r])

# img[0:100, 100:200, 0] = 255
# img[50,100] = (0,0,255)

# cv2.namedWindow("img", cv2.WINDOW_NORMAL)
# cv2.imshow("img", img)

cv2.imshow("Blue", b)
cv2.imshow("Green", g)
cv2.imshow("Red", r)

cv2.imshow("Merged", merged)

cv2.waitKey()
