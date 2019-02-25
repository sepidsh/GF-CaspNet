import sys
import os
sys.path.append(os.curdir)
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.DEBUG)
import cv2
import detect_face as FD2
import numpy as np
import math


def drawBoxes(im, boxes):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    for i in range(x1.shape[0]):
        cv2.rectangle(im, (int(x1[i]), int(y1[i])), (int(x2[i]), int(y2[i])), (0, 255, 0), 1)
    return im
def get_webcam_image(cam):

    ret_val, img = cam.read()
    img=cv2.flip(img,1)

    return img

def show_age(Bbox,age,gender,img):
    x=Bbox[:,0]
    y=Bbox[:,1]
    if len(age)> len(gender):
        iteration=len(gender)
    else:
        iteration=Bbox.shape[0]

    for face_num in range(iteration):
        if age[face_num]==200 or gender[face_num]==200:
            img = cv2.putText(img, 'unknown', (int(x[face_num]), int(y[face_num])),
                              cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0))
        else:
            if gender[face_num]==1:
                g='Male'
            else:
                g='Female'
            img = cv2.putText(img, 'A: ' + "%.1f" % age[face_num], (int(x[face_num]), int(y[face_num])-5), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0))
            img = cv2.putText(img, 'G: ' + g, (int(x[face_num]), int(y[face_num])-30),
                              cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 255, 0))
    return img
def rotate_image(mat, angle):
    height, width = mat.shape[:2]
    image_center = (width / 2, height / 2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1)

    radians = math.radians(angle)
    sin = math.sin(radians)
    cos = math.cos(radians)
    bound_w = int((height * abs(sin)) + (width * abs(cos)))
    bound_h = int((height * abs(cos)) + (width * abs(sin)))

    rotation_mat[0, 2] += ((bound_w / 2) - image_center[0])
    rotation_mat[1, 2] += ((bound_h / 2) - image_center[1])

    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat
def markLandmark(im, points):
    """
    Mark landmarks on the image.

    Args:
        points: [#faces][5landmarks]
        landmarks_col(x): 0~4 index 
        landmarks_row(y): 5~9 index
    Returns:
        returns image 
    """
    for i in range(points.shape[0]):
        for j in range(int(points.shape[1] / 2)):
            cv2.circle(im, (int(points[i][j]), int(points[i][j + 5])), 1, (0, 255, 0))
    return im
def main():

    minsize = 20
    threshold = [0.6, 0.7, 0.7]
    factor = 0.709


    print('Creating networks and loading parameters')


    sess = tf.Session()



    with tf.variable_scope("MTCNN"):

        pnet, rnet, onet = FD2.create_mtcnn(sess, None,"MTCNN")
    writer = tf.summary.FileWriter("./graphs", sess.graph)


    img = cv2.imread('./test1.jpg')
    img_matlab = img.copy()

    # BGR -> RGB
    tmp = img_matlab[:, :, 2].copy()
    img_matlab[:, :, 2] = img_matlab[:, :, 0]
    img_matlab[:, :, 0] = tmp

    boundingboxes, points = FD2.detect_face(img_matlab, minsize, pnet, rnet, onet, threshold, factor)
    points=np.reshape(points.T, (-1,10))

    img = markLandmark(img,points)
    img = drawBoxes(img, boundingboxes)

    cv2.imshow('window',img)
    cv2.waitKey(0)



    sess.close()


if __name__ == '__main__':
    main()
