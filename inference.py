# https://github.com/Kazuhito00/MoveNet-Python-Example/blob/main/MoveNet_tf2onnx.ipynb
import onnxruntime as rt
import cv2
import numpy as np
import time

onnxnet = rt.InferenceSession('singlepose-thunder.onnx')
input_detail = onnxnet.get_inputs()
output_detail = onnxnet.get_outputs()

print(len(input_detail), len(output_detail))
print(input_detail[0])
print(output_detail[0])
# Set up video capture
cap = cv2.VideoCapture(0)  # 0 for default camera

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # El modelo thunder necesita imagenes de 256x256
    img = cv2.resize(frame, (256, 256))
    
    # Ensure image has 3 channels (RGB)
    if img.shape[2] != 3:
        raise ValueError(f"Invalid number of channels in input image: {img.shape[2]}. Expected 3 channels.")
    
    # This is needed for ONNX model input format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    # el modelo necesita un que los valore de int32
    imatge_array = np.array(img, dtype=np.int32)
    
    # Añade una dimensión extra al principio del array para representar el batch size (1)
    # Convierte la imagen de forma [256, 256, 3] a [1, 256, 256, 3] para cumplir con el formato
    # de entrada que espera el modelo (batch, height, width, channels)
    tensor_imatge = np.expand_dims(imatge_array, axis=0)
    
    # Prepare the input for the ONNX model

    inputs = {onnxnet.get_inputs()[0].name: tensor_imatge}

    
    # # Run inference
    
    start_time = time.time()
    outputs = onnxnet.run(None, inputs)
    inference_time = time.time() - start_time
    print(f"Inference time: {inference_time:.4f} seconds")

    # Process the output to get keypoints
    keypoints = outputs[0][0][0]  # Shape: [17, 3] - 17 keypoints with [y, x, confidence]
    
    # Define keypoint connections for visualization (COCO format)
    KEYPOINT_CONNECTIONS = [
        (0, 1),    # nose to left_eye
        (0, 2),    # nose to right_eye
        (1, 3),    # left_eye to left_ear
        (2, 4),    # right_eye to right_ear
        (0, 5),    # nose to left_shoulder
        (0, 6),    # nose to right_shoulder
        (5, 7),    # left_shoulder to left_elbow
        (7, 9),    # left_elbow to left_wrist
        (6, 8),    # right_shoulder to right_elbow
        (8, 10),   # right_elbow to right_wrist
        (5, 6),    # left_shoulder to right_shoulder
        (5, 11),   # left_shoulder to left_hip
        (6, 12),   # right_shoulder to right_hip
        (11, 12),  # left_hip to right_hip
        (11, 13),  # left_hip to left_knee
        (13, 15),  # left_knee to left_ankle
        (12, 14),  # right_hip to right_knee
        (14, 16)   # right_knee to right_ankle
    ]
    
    # Create a copy of the original frame for drawing
    display_frame = frame.copy()
    
    # Scale keypoints to original frame dimensions
    h, w = frame.shape[:2]
    scaled_keypoints = []
    
    for kp in keypoints:
        y, x, conf = kp
        # Scale coordinates from 0-1 to image dimensions
        x_scaled = int(x * w)
        y_scaled = int(y * h)
        scaled_keypoints.append((x_scaled, y_scaled, conf))
    
    # Draw keypoints and connections
    for idx, (x, y, conf) in enumerate(scaled_keypoints):
        if conf > 0.3:  # Only draw keypoints with confidence above threshold
            cv2.circle(display_frame, (x, y), 5, (0, 255, 0), -1)
    
    # Draw connections between keypoints
    for connection in KEYPOINT_CONNECTIONS:
        start_idx, end_idx = connection
        if (scaled_keypoints[start_idx][2] > 0.3 and 
            scaled_keypoints[end_idx][2] > 0.3):
            start_point = (scaled_keypoints[start_idx][0], scaled_keypoints[start_idx][1])
            end_point = (scaled_keypoints[end_idx][0], scaled_keypoints[end_idx][1])
            cv2.line(display_frame, start_point, end_point, (0, 0, 255), 2)
    
    # Update frame to display the annotated version
    frame = display_frame
    


    # Display the resulting frame
    cv2.imshow('Frame', frame)
    
    # Break the loop on 'q' key press


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources when done
cap.release()
cv2.destroyAllWindows()     
