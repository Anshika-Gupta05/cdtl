import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Multi-Task DenseNet Model
class MultiTaskDenseNet(nn.Module):
    def __init__(self):
        super(MultiTaskDenseNet, self).__init__()
        self.backbone = models.densenet121(pretrained=False)
        num_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()

        self.pneumonia_head = nn.Linear(num_features, 2)
        self.tb_head = nn.Linear(num_features, 2)
        self.fibrosis_head = nn.Linear(num_features, 2)

    def forward(self, x):
        features = self.backbone(x)
        return (
            self.pneumonia_head(features),
            self.tb_head(features),
            self.fibrosis_head(features)
        )

# Load model
def load_model(model_path):
    model = MultiTaskDenseNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Prediction
def predict_image(image, model):
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        pneumonia_out, tb_out, fibrosis_out = model(input_tensor)
        softmax = nn.Softmax(dim=1)

        pneumonia_probs = softmax(pneumonia_out)
        tb_probs = softmax(tb_out)
        fibrosis_probs = softmax(fibrosis_out)

        pneumonia_pred = torch.argmax(pneumonia_probs, dim=1).item()
        tb_pred = torch.argmax(tb_probs, dim=1).item()
        fibrosis_pred = torch.argmax(fibrosis_probs, dim=1).item()

        label_map = {0: "Normal", 1: "Disease"}

        return {
            "Pneumonia": {
                "label": label_map[pneumonia_pred],
                "confidence": round(pneumonia_probs[0][pneumonia_pred].item(), 3)
            },
            "Tuberculosis": {
                "label": label_map[tb_pred],
                "confidence": round(tb_probs[0][tb_pred].item(), 3)
            },
            "Fibrosis": {
                "label": label_map[fibrosis_pred],
                "confidence": round(fibrosis_probs[0][fibrosis_pred].item(), 3)
            }
        }

# Grad-CAM Utility
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, target_idx, output):
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][target_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        gradients = self.gradients[0].detach().cpu().numpy()
        activations = self.activations[0].detach().cpu().numpy()
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam -= cam.min()
        cam /= cam.max() if cam.max() > 0 else 1
        return cam

# Apply Grad-CAM to all diseases
def apply_gradcam(image, model):
    model.eval()
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    target_layer = model.backbone.features.denseblock3.denselayer16.conv2
    gradcam = GradCAM(model, target_layer)

    outputs = model(input_tensor)
    diseases = ['Pneumonia', 'Tuberculosis', 'Fibrosis']
    heatmaps = {}

    for i, output in enumerate(outputs):
        pred_class = torch.argmax(output, dim=1).item()
        cam = gradcam.generate(input_tensor, pred_class, output)

        heatmap = np.uint8(255 * cam)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        original = cv2.cvtColor(np.array(image.resize((224, 224))), cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
        heatmaps[diseases[i]] = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

    return heatmaps
