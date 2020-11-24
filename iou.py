def iou(rect1,rect2):
    '''
    Calculate the intersection ratio of two rectangles
    : param rect1: the first rectangle. Denoted by X, y, W, h, where x, y are the coordinates of the upper right corner of the rectangle
    : param rect2: the second rectangle.
    : Return: returns intersection ratio, that is, intersection ratio Union
    '''
    x1,y1,w1,h1=rect1
    x2,y2,w2,h2=rect2
    
    inter_w=(w1+w2)-(max(x1+w1,x2+w2)-min(x1,x2))
    inter_h=(h1+h2)-(max(y1+h1,y2+h2)-min(y1,y2))
    
    if inter_h<=0 or inter_w <= 0: 
        return 0
    inter=inter_w * inter_h
    
    union=w1*h1+w2*h2-inter
    return inter/union


