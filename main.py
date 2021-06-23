#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run a YOLO_v3 style detection model on test images.
"""
import numpy as np
import argparse

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.general import (
    check_img_size, non_max_suppression, scale_coords, plot_one_box, )
from utils.torch_utils import select_device
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *

flag = False


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


# with torch.no_grad():
#     detect()
class Ui_MainWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)
        self.timer_camera = QtCore.QTimer()
        self.timer_camera_capture = QtCore.QTimer()
        self.cap = cv2.VideoCapture()
        self.CAM_NUM = 0
        self.set_ui()
        self.slot_init()
        # self.detect_image(self.image)
        self.__flag_work = 0
        self.x = 0
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default='weights/last.pt', help='model.pt path(s)')
        parser.add_argument('--source', type=str, default='images', help='source')  # file/folder, 0 for webcam
        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.45, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='display results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--save-dir', type=str, default='results', help='directory to save results')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--update', action='store_true', help='update all models')
        self.opt = parser.parse_args()
        print(self.opt)
        ut, source, weights, view_img, save_txt, imgsz = \
            self.opt.save_dir, self.opt.source, self.opt.weights, self.opt.view_img, self.opt.save_txt, self.opt.img_size
        self.device = select_device(self.opt.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        self.imgsz = check_img_size(imgsz, s=self.model.stride.max())  # check img_size
        if self.half:
            self.model.half()  # to FP16

        cudnn.benchmark = True  # set True to speed up constant image size inference

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]

    def set_ui(self):

        self.__layout_main = QtWidgets.QVBoxLayout()
        self.__layout_fun_button = QtWidgets.QHBoxLayout()
        self.__layout_data_show = QtWidgets.QVBoxLayout()

       

        self.openimage = QtWidgets.QPushButton(u'Select an image')
        self.opencameras = QtWidgets.QPushButton(u'Open the front-facing camera')
        self.train = QtWidgets.QPushButton(u'Select a video')


        #颜色
        self.openimage.setStyleSheet('''QPushButton{font-family:'Times New Roman';font-size:15px;background:#cfcfcf;border-radius:10px;}QPushButton:hover{background:#5c5c5c;}''')
        self.opencameras.setStyleSheet('''QPushButton{font-family:'Times New Roman';font-size:15px;background:#cfcfcf;border-radius:10px;}QPushButton:hover{background:#5c5c5c;}''')
        self.train.setStyleSheet('''QPushButton{font-family:'Times New Roman';font-size:15px;background:#cfcfcf;border-radius:10px;}QPushButton:hover{background:#5c5c5c;}''')
       
        # self.Openvideo = QtWidgets.QPushButton(u'Select a video')
        self.openimage.setMinimumHeight(40)
        self.opencameras.setMinimumHeight(40)
        self.train.setMinimumHeight(40)
        # self.Openvideo.setMinimumHeight(50)

        self.openimage.move(50,50)
        #self.openimage.move(10, 30)
        self.opencameras.move(100,100)
        #self.opencameras.move(10, 50)
        self.train.move(150,150)
        #self.train.move(70, 30)

    



        # 信息显示
        self.showimage = QtWidgets.QLabel()

        self.showimage.setFixedSize(641, 481)
        self.showimage.setAutoFillBackground(False)

        self.__layout_fun_button.addWidget(self.openimage)
        self.__layout_fun_button.addWidget(self.opencameras)
        self.__layout_fun_button.addWidget(self.train)
        # self.__layout_fun_button.addWidget(self.Openvideo)

        self.__layout_main.addLayout(self.__layout_fun_button)
        self.__layout_main.addWidget(self.showimage)

        self.setLayout(self.__layout_main)
        # self.label_move.raise_()
        self.setWindowTitle(u'Safety Helmet detection (Demo)')

    def slot_init(self):
        self.openimage.clicked.connect(self.button_open_image_click)
        self.opencameras.clicked.connect(self.button_opencameras_click)
        self.timer_camera.timeout.connect(self.show_camera)
        # self.timer_camera_capture.timeout.connect(self.capture_camera)
        self.train.clicked.connect(self.button_train_click)
        # self.Openvideo.clicked.connect(self.Openvideo_click)

    def button_open_image_click(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "Select an image", "", "*.jpg;;*.png;;All Files(*)")
        img = cv2.imread(imgName)
        print(imgName)
        showimg = img
        with torch.no_grad():
            img = letterbox(img, new_shape=self.opt.img_size)[0]
            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            # Inference
            pred = self.model(img, augment=self.opt.augment)[0]
            # Apply NMS
            pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                       agnostic=self.opt.agnostic_nms)
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], showimg.shape).round()

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        label = '%s %.2f' % (self.names[int(cls)], conf)
                        plot_one_box(xyxy, showimg, label=label, color=self.colors[int(cls)], line_thickness=1)
        self.result = cv2.cvtColor(showimg, cv2.COLOR_BGR2BGRA)
        self.result = cv2.resize(self.result, (640, 480), interpolation=cv2.INTER_AREA)
        self.QtImg = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                  QtGui.QImage.Format_RGB32)
        # 显示图片到label中;
        self.showimage.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))

    def button_train_click(self):
        global flag
        self.timer_camera_capture.stop()
        self.cap.release()
        if flag == False:
            flag = True
            imgName, imgType = QFileDialog.getOpenFileName(self, "Select a video", "", "*.mp4;;*.avi;;All Files(*)")
            flag = self.cap.open(imgName)
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"Fail to select a video",
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.timer_camera.start(30)
                self.train.setText(u'Close the detection')
        else:
            flag = False
            self.timer_camera.stop()
            self.cap.release()
            self.showimage.clear()
            self.train.setText(u'Select a video')

    def button_opencameras_click(self):
        self.timer_camera_capture.stop()
        self.cap.release()
        if self.timer_camera.isActive() == False:
            flag = self.cap.open(self.CAM_NUM)
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"Please check if the camera is correctly connected to the computer",
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.timer_camera.start(30)

                self.opencameras.setText(u'Close the detection')
        else:
            self.timer_camera.stop()
            self.cap.release()
            self.showimage.clear()
            self.opencameras.setText(u'Open the front-facing camera')

    def show_camera(self):
        flag, img = self.cap.read()
        if img is not None:
            showimg = img
            with torch.no_grad():
                img = letterbox(img, new_shape=self.opt.img_size)[0]
                # Convert
                img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                # Inference
                pred = self.model(img, augment=self.opt.augment)[0]

                # Apply NMS
                pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                           agnostic=self.opt.agnostic_nms)
                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], showimg.shape).round()
                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            label = '%s %.2f' % (self.names[int(cls)], conf)
                            print(label)
                            plot_one_box(xyxy, showimg, label=label, color=self.colors[int(cls)], line_thickness=3)
            show = cv2.resize(showimg, (640, 480))
            self.result = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
            showImage = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                     QtGui.QImage.Format_RGB888)
            self.showimage.setPixmap(QtGui.QPixmap.fromImage(showImage))
        else:
            flag = False
            self.timer_camera.stop()
            self.cap.release()
            self.showimage.clear()
            self.train.setText(u'Select a video')


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    ui.show()
    sys.exit(app.exec_())
