def compute_iou(rec1, rec2):
    """
    computing IoU
     IOU = 相交的面积 /（两个框的面积和 - 相交的面积
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """

    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return intersect / (sum_area - intersect)
'''
label1 = predicted_class,score, top, left, bottom, right,c
'''
threshold = 0.3
def get_conincode_f(label_list,i):
    global threshold
    # 重复的框
    coincide_f = []
    # if min_val< 阈值 将其放入coincide_f ，
    for label1 in label_list:
        for label2 in label_list:
            if label2 == label1:
                continue
            rec1, rec2 = label1[2:6],label2[2:6]
            iou = compute_iou(rec1, rec2)
            # print (iou)
            # print (label1[1])
            # print (label2[1])
            if iou > threshold:
                if label1[1] < label2[1] and label1 not in coincide_f:
                    coincide_f.append(label1)
                elif label1[1] > label2[1] and label2 not in coincide_f:
                    coincide_f.append(label2)

    return coincide_f


def del_f_useIOU(label_list):
    coincide_f = get_conincode_f(label_list, 2)
    namelist = get_labelName(label_list)
    if len(coincide_f) == 0:
        return label_list,namelist
    else:
        for f in coincide_f:
            # print (f)
            label_list.remove(f)
            namelist.remove(f[0])
        return label_list,namelist

def get_labelName(label_list):
    nameList = []
    for label in label_list:
        nameList.append(label[0])
    return nameList
