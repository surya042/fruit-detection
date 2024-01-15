import cv2
import numpy as np
import time 

cap = cv2.VideoCapture('videos/123.mp4') 
path_label = 'coco.names' 
weight_height_target = 320 
confThreshold = 0.5 
nmsThreshold = 0.4 
inccount1 = 0 
inccount2 = 0 
inccount3 = 0 
inccount4 = 0 
inccount5 = 0 
inccount_reset = 0 
start_time = time.time() 

net = cv2.dnn.readNet("yolov4.cfg","yolov4.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV) 
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU) 

result = cv2.VideoWriter('result.avi',cv2.VideoWriter_fourcc(*'XVID'),20,(320,320)) 

classes = [] 
with open("coco.names","rt") as f: 
    classes = [line.strip() for line in f.readlines()] 
layers_name = net.getLayerNames() 
output_layers = [layers_name[i-1] for i in net.getUnconnectedOutLayers()] 
colors = np.random.uniform(0,255,size=(len(classes),3))

def findObject(ouputs,frame): 
    height,width,channel = frame.shape 
    boxes = [] 
    confs = [] 
    class_ids = [] 
    count1 = 0 
    count2 = 0 
    count3 = 0 
    count4 = 0 
    count5 = 0 
    
    for output in outputs: 
        for detect in output: 
            scores = detect[5:] 
            class_id = np.argmax(scores) 
            conf = scores[class_id] 
            if conf > confThreshold: 
               center_x = int(detect[0] * width) 
               center_y= int(detect[1] * height) 
               w = int(detect[2] * width) 
               h = int(detect[3] * height) 
               x= int(center_x - w/2) 
               y= int(center_y - h/2) 
               boxes.append([x, y, w, h]) 
               confs.append(float(conf)) 
               class_ids.append(class_id) 
            else: 
               continue 
    draw_box = cv2.dnn.NMSBoxes(boxes, confs, confThreshold, nmsThreshold) 
    font = cv2.FONT_HERSHEY_PLAIN 
    for i in draw_box: 
        i=i
        box = boxes[i] 
        label = str(classes[class_ids[i]]) 
        color =colors[i] 
        x,y,w,h = box[0], box[1], box[2], box[3] 
        xMid = int((x+(x+w))/2) 
        yMid = int((y+(y+h))/2) 
        if yMid > 10 and yMid < 300 and xMid > 213 and xMid < 217: 
            if class_ids[i] == 46: 
                count1 = count1 + 1 
            elif class_ids[i] == 47: 
                count2 = count2 + 1 
            elif class_ids[i] == 49: 
                count3 = count3 + 1 
            elif class_ids[i] == 50: 
                count4 = count4 + 1 
            elif class_ids[i] == 41: 
                count5 = count5 + 1 
        cv2.circle(frame, (xMid, yMid), 1, (0, 0, 255), 2) 
        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 1) 
        cv2.putText(frame, label, (x,y-5), font, 0.3, color, 1) 
        cv2.line(frame, (220, 10), (220, 300), (0, 0, 255, 3)) 
    return count1, count2, count3, count4, count5 

while True: 
    _,frame =cap.read() 
    frame = cv2.resize(frame, (weight_height_target, weight_height_target)) 
    blob = cv2.dnn.blobFromImage(frame, scalefactor=0.00392, size=(320,320), mean=(0, 0, 0), swapRB=True, crop=False )
    net.setInput(blob) 
    outputs = net.forward(output_layers)
    counter1, counter2, counter3, counter4, counter5 = findObject(outputs, frame) 
    inccount1 = inccount1 + counter1 
    inccount2 = inccount2 + counter2 
    inccount3 = inccount3 + counter3 
    inccount4 = inccount4 + counter4 
    inccount5 = inccount5 + counter5 
    run_time = time.time() 
    inccount_reset = int(time.time() - start_time) 
    if inccount_reset == 3600: 
        inccount1 = 0 
        inccount2 = 0 
        inccount3 = 0 
        inccount4 = 0 
        inccount5 = 0 
        inccount_reset = 0 
        start_time = run_time 
        cv2.putText(frame, f'Banana: {inccount1}', (5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1) 
        cv2.putText(frame, f'Apple: {inccount2}', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1) 
        cv2.putText(frame, f'Orange: {inccount3}', (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1) 
        cv2.putText(frame, f'Broccoli: {inccount4}', (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1) 
        cv2.putText(frame, f'Carrot: {inccount5}', (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1) 
        result.write(frame) 
        cv2.imshow('video',frame)
        
        if cv2.waitkey(1) & 0xFF == ord('q'): 
            break 
        
    cap.release() 
    result.release() 
    cv2.destroyAllwindows()
        
        
        
        
        
        