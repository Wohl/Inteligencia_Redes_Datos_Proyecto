import numpy as np
import cv2
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn

class ObstacleConfig:
    def __init__(self):
        self.num_classes = 2
        self.score_threshold = 0.9
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def detect_obstacles(image_path):
    # Cargar la imagen
    image = cv2.imread("imagen.jpg")

    # Cargar el Modelo
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=2)
    model.load_state_dict(torch.load('resnet50-19c8e357.pth'))

    # Setter el modo de evalucion del modelo
    model.eval()

    # Convertir la imagen a PyTorchTensor
    image_tensor = torchvision.transforms.functional.to_tensor(image)

    # Pass the image through the model
    outputs = model([image_tensor])

    # Extraer la prediccion de mascaras y alturas
    masks = outputs[0]['masks'].detach().numpy()
    heights = outputs[0]['depths'].detach().numpy()

    # Return las mascaras y alturas
    return masks, heights


def get_obstacle_height(obstacle_mask):
    # Calcular la altura de un obstaculo desde la mascara
    height = np.sum(obstacle_mask) / np.sum(obstacle_mask > 0)
    return height
