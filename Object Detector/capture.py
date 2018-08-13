import cv2

cap = cv2.VideoCapture(0) # 0 stands for very first webcam attach
filename='output.avi'
codec=cv2.VideoWriter_fourcc('m','p','4','v')#fourcc stands for four character code
framerate=30
resolution=(640,480)
    
VideoFileOutput=cv2.VideoWriter(filename,codec,framerate, resolution)
    
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    
    ret=True
        
    
    while (ret):
        
        
        ret, image_np=cap.read() 


        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
       
          # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
          # Actual detection.
        (boxes, scores, classes, num) = sess.run(
              [detection_boxes, detection_scores, detection_classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})
          # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              line_thickness=8)
  
        VideoFileOutput.write(image_np)
        cv2.imshow('live_detection',image_np)
        if cv2.waitKey(25) & 0xFF==ord('q'):
            break
            cv2.destroyAllWindows()
            cap.release()
