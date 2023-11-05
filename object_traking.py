import cv2
import numpy as np

# Function for image classification
def classify_image(image):
    # Preprocess the image
    blob = cv2.dnn.blobFromImage(image, 0.017, (224, 224), (103.94, 116.78, 123.68))

    # Read class labels
    class_labels = open('densenet_121/synset_words.txt').read().strip().split("\n")
    image_classes = [label[label.find(" ") + 1:].split(",")[0] for label in class_labels]

    # Load the pre-trained model
    model = cv2.dnn.readNetFromCaffe('densenet_121/DenseNet_121.prototxt', 'densenet_121/DenseNet_121.caffemodel')
    model.setInput(blob)
    
    # Forward pass
    output = model.forward()
    predictions = output.reshape(len(output[0][:]))

    # Calculate probabilities
    exp_predictions = np.exp(predictions - np.max(predictions))
    probabilities = exp_predictions / exp_predictions.sum()
    confidence = np.max(probabilities)
    class_index = np.argmax(probabilities)
    class_name = image_classes[class_index]
    result_text = "{}: {:.2f}%".format(class_name, confidence * 100)

    return result_text

# Function for tracking
def object_tracking():
    # Initialize the tracker (choose one of the trackers)
    tracker = cv2.TrackerCSRT_create()
    
    # Open the video camera
    video_capture = cv2.VideoCapture(0)

    while True:
        _, frame = video_capture.read()

        cv2.imshow("Object Tracking", frame)

        key = cv2.waitKey(30) & 0xff
        if key == 27:  # Exit if 'Esc' key is pressed
            break

    # Select a region of interest (ROI)
    bounding_box = cv2.selectROI(frame, False)

    is_initialized = tracker.init(frame, bounding_box)
    cv2.destroyWindow("ROI selector")

    while True:
        is_tracking, frame = video_capture.read()
        is_tracking, bounding_box = tracker.update(frame)

        if is_tracking:
            target_image = crop_image(frame, bounding_box)

            text = classify_image(target_image)

            frame = draw_bounding_box(text, bounding_box, frame)

        cv2.imshow("Object Tracking", frame)

        key = cv2.waitKey(1) & 0xff
        if key == 27:  # Exit if 'Esc' key is pressed
            break

# Function to crop an image based on a bounding box
def crop_image(image, bounding_box):
    height, width, _ = image.shape
    x, y, w, h = bounding_box

    new_x = x - w * 0.3
    new_y = y - h * 0.3
    new_w = w * 1.6
    new_h = h * 1.6

    cropped_image = image[max(int(new_y), 0):min(int(new_y + new_h), height - 1),
                          max(int(new_x), 0):min(int(new_x + new_w), width - 1)]
    cv2.imshow("ROI", cropped_image)
    cv2.waitKey(1)
    return cropped_image

# Function to draw a bounding box and label on the frame
def draw_bounding_box(text, bounding_box, frame):
    x, y, w, h = bounding_box
    font = cv2.FONT_HERSHEY_SIMPLEX
    label_size = cv2.getTextSize(text, font, 1, 2)
    text_origin = np.array([bounding_box[0], bounding_box[1] - label_size[0][1]])

    cv2.rectangle(frame, tuple(text_origin), tuple(text_origin + label_size[0]),
                  color=(0, 0, 255), thickness=-1)

    cv2.putText(frame, text, (bounding_box[0], bounding_box[1] - 5), font, 1, (255, 255, 255), 2)

    cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2, 2)

    return frame

if __name__ == "__main__":
    object_tracking()
