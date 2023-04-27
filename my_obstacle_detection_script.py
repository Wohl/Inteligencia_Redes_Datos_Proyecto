import cv2
from my_obstacle_detection_module import ObstacleConfig, detect_obstacles

# Crear una instancia de ObstacleConfig
config = ObstacleConfig()

# Especificar la ruta de la imagen
image_path = 'imagen.jpg'

# Cargar la imagen
image = cv2.imread(image_path)

# Detectar los obstáculos en la imagen
obstacle_masks, obstacle_heights = detect_obstacles(image_path)
