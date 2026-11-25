import cv2
import supervision as sv
from rfdetr import RFDETRNano, RFDETRMedium
import matplotlib.pyplot as plt

model = RFDETRNano()
image = cv2.imread("./images/Terrain.png", cv2.IMREAD_COLOR_RGB)
detections = model.predict(image)

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

annotated_image = box_annotator.annotate(scene=image, detections=detections)
annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

plt.figure()
plt.imshow(annotated_image)
plt.show()