#! /usr/bin/env python

import os
import argparse
import json
import cv2
from utils.utils import get_yolo_boxes
from keras.models import load_model

NET_HEIGHT, NET_WIDTH = 480, 640  # a multiple of 32, the smaller the faster
OBJ_THRESH, NMS_THRESH = 0.3, 0.45


def find_gate_in(boxes, label):
    print("Gate found. calculating center")
    gate_boxes = [box for box in boxes if label in box.classes]
    if len(gate_boxes) == 0: return None
    x_center = sum([b.xmin + b.xmax for b in gate_boxes]) / (len(gate_boxes)*2)
    y_center = sum([b.ymin + b.ymax for b in gate_boxes]) / (len(gate_boxes)*2)
    return x_center, y_center


def _main_(args):
    anchors, labels, weights = config_params(args)
    infer_model = load_model(weights)
    video_reader = cv2.VideoCapture(0)
    gate_label = [index for label, index in enumerate(labels) if label == "gate"][0]
    print("Gate label is: " + gate_label)
    while True:
        print("Reading image")
        ret_val, image = video_reader.read()
        if not ret_val: continue
        print("Finding gate")
        yolo_boxes = get_yolo_boxes(infer_model, [image], NET_HEIGHT, NET_WIDTH, anchors, OBJ_THRESH, NMS_THRESH)
        gate_location = find_gate_in(yolo_boxes, gate_label)
        if gate_location is not None: print(gate_location)


def config_params(args):
    with open(args.conf) as config_buffer:
        config = json.load(config_buffer)
    anchors = config['model']['anchors']
    labels = config['model']['labels']
    weights = config['train']['saved_weights_name']
    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
    return anchors, labels, weights


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="Run Poseidon's vision with a trained YOLO model")
    arg_parser.add_argument('-c', '--conf', default="./config.json", help='path to configuration file')

    _main_(arg_parser.parse_args())
