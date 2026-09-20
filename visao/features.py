import torch
import torch.nn as nn
from torchvision import models
from PIL import Image


class FeatureExtractor:

    def __init__(self):

        # Escolhe GPU se existir, caso contrário usa CPU
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print("Dispositivo:", self.device)

        # Carrega o MobileNetV3 Small
        weights = models.MobileNet_V3_Small_Weights.DEFAULT

        model = models.mobilenet_v3_small(
            weights=weights
        )

        # Pegamos somente a parte que extrai características
        self.model = nn.Sequential(
            model.features,
            model.avgpool,
            nn.Flatten()
        )

        self.model.eval()
        self.model.to(self.device)

        # Pré-processamento oficial do modelo
        self.transform = weights.transforms()


    def extract(self, image):

        # OpenCV usa BGR.
        # PIL normalmente trabalha com RGB.
        image = image[:, :, ::-1]

        # Converte para imagem PIL
        image = Image.fromarray(image)

        # Redimensionamento e normalização
        image = self.transform(image)

        # Adiciona dimensão do batch
        image = image.unsqueeze(0)

        # Envia para CPU ou GPU
        image = image.to(self.device)

        # Extrai as características
        with torch.no_grad():

            features = self.model(image)

            # Normaliza o vetor
            features = torch.nn.functional.normalize(
                features,
                p=2,
                dim=1
            )

        # Volta para NumPy
        return features.cpu().numpy()[0]