test_image = plt.imread(os.path.join('test_images', 'test4.jpg'))
undistorted_img = cv2.undistort(test_image, mtx, dist)

thresh_binary = func(image)


img_size = (thresh_binary.shape[1], thresh_binary.shape[0])
width, height = img_size
offset = 200
src = np.float32([
    [  588,   446 ],
    [  691,   446 ],
    [ 1126,   673 ],
    [  153 ,   673 ]])
dst = np.float32([[offset, 0], [img_size[0] - offset, 0], [img_size[0] - offset, img_size[1]], [offset, img_size[1]]])
M = cv2.getPerspectiveTransform(src,dst)
Minv = cv2.getPerspectiveTransform(dst, src)

plt.figure(figsize=(10,40))
plt.subplot(1,2,1)
thresh_binary = func(image)
plt.imshow(thresh_binary, cmap='gray')
plt.title('Thresholded Binary')

plt.subplot(1,2,2)
binary_warped = cv2.warpPerspective(thresh_binary,M, (width, height))
plt.imshow(binary_warped, cmap='gray')
plt.title('Binary Warped Image')