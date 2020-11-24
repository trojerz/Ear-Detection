import cv2, sys

def get_coordinates(pic, path):
    #returns coordinates of ear in a picture from a mask. 
    img_path = "datasets\\AWEForSegmentation\\" + path + "\\" + str(pic) + ".png"
 
    image = cv2.imread(img_path,0)

    cv2.imshow('Input',image)
 
    _, bin_img = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)
    cv2.imshow('Intermediate',bin_img)
    _, contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points = []
    for i in range(len(contours)):#range(len(contours)):
        x_coordinates = []
        y_coordinates = []
        for j in range(4):
            x_coordinates.append(contours[i][j][0][0])
            y_coordinates.append(contours[i][j][0][1])
        points.append([min(x_coordinates), min(y_coordinates), max(x_coordinates),  max(y_coordinates)])
    return(points, len(contours))

if __name__ == "__main__" :
    get_coordinates(sys.argv[0], 'trainannot_rect')