# 캘리브레이션 영상을 가지고 정보 구하여 dump로 저장하기
import cv2
import numpy as np
import glob
import pickle

x = 9
y = 6

objp = np.zeros((y * x, 3), np.float32)
objp[:, :2] = np.mgrid[0:x, 0:y].T.reshape(-1, 2)

objpoints = []
imgpoints = []

# 사용자가 제시한 조건에 맞는 파일명을 리스트 형식으로 반환
images = glob.glob('cal_img/img*.jpg')

total_images = len(images)
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (x, y), None)

    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        cv2.drawChessboardCorners(img, (x, y), corners, ret)

        write_name = 'output/corners_found' + str(idx) + '.jpg'
        cv2.imwrite(write_name, img)

        out_str = f'{idx}/{total_images}'
        cv2.putText(img, out_str, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

img = cv2.imread('cal_img/img8.jpg')
height, width = img.shape[:2]
img_size = (width, height)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

dst = cv2.undistort(img, mtx, dist, None, mtx)

cv2.imwrite('output/test_undist.jpg', dst)

dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump(dist_pickle, open('output/wide_dist_pickle.p', 'wb'))
print('mtx', mtx)
print('dist', dist)

img_result = cv2.hconcat([img, dst])
img_result = cv2.pyrDown(img_result)

cv2.imshow('dst', img_result)
cv2.waitKey(0)
cv2.destroyAllWindows()