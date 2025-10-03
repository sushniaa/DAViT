import torch
import numpy as np
from pytorch_grad_cam import HiResCAM, LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.metrics.road import ROADMostRelevantFirst
from shap import DeepExplainer
import matplotlib.pyplot as plt

def denormalize(img_tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * std + mean
    return np.clip(img, 0, 1)

class ModelWithDomain(nn.Module):
    def __init__(self, model, domain_id):
        super().__init__()
        self.model = model
        self.domain_id = domain_id

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if self.domain_id is None:
            default_domain = torch.tensor([9], device=x.device)
            return self.model(x, default_domain)
        return self.model(x, self.domain_id)

def apply_xai(model, dataloader, num_samples=5, class_names=None, ontology=None, domain_map=None):
    model.eval()
    device = next(model.parameters()).device
    targets = [ClassifierOutputTarget(i) for i in range(9)]
    road_metric = ROADMostRelevantFirst(percentile=75)
    target_layers_hires = [model.vit.encoder.layer[-1].layernorm_before]
    target_layers_layer = [model.vit.encoder.layer[0].layernorm_before]

    sample_count = 0
    for batch in dataloader:
        images, labels, domains = batch
        images = images.to(device)
        labels = labels.to(device)
        domains = domains.to(device)

        for i in range(len(images)):
            if sample_count >= num_samples:
                break

            image = images[i]
            label = labels[i].item()
            domain = domains[i].item()
            input_tensor = image.unsqueeze(0).to(device)
            domain_id = torch.tensor([domain], device=device)
            rgb_img = denormalize(image)

            model_with_domain = ModelWithDomain(model, domain_id)
            model_no_domain = ModelWithDomain(model, None)

            with HiResCAM(model=model_with_domain, target_layers=target_layers_hires, reshape_transform=vit_reshape_transform) as cam:
                hires_cam = cam(input_tensor=input_tensor, targets=targets, aug_smooth=True)
                pred_idx = torch.argmax(model_with_domain(input_tensor), dim=1).item()
                hires_vis = show_cam_on_image(rgb_img, hires_cam[0], use_rgb=True)

            with LayerCAM(model=model_with_domain, target_layers=target_layers_layer, reshape_transform=vit_reshape_transform) as cam:
                layer_cam = cam(input_tensor=input_tensor, targets=targets, aug_smooth=True)
                layer_vis = show_cam_on_image(rgb_img, layer_cam[0], use_rgb=True)

            with HiResCAM(model=model_no_domain, target_layers=target_layers_hires, reshape_transform=vit_reshape_transform) as cam:
                no_domain_cam = cam(input_tensor=input_tensor, targets=targets, aug_smooth=True)
                diff_cam = hires_cam[0] - no_domain_cam[0]
                diff_vis = show_cam_on_image(rgb_img, diff_cam, use_rgb=True)

            road_scores = road_metric(input_tensor, hires_cam, targets, model_with_domain)
            road_score_mean = road_scores.mean() if road_scores.size else 0.0

            pred_class = class_names[pred_idx]
            true_class = class_names[label]
            features_detected = []
            cam_max = hires_cam[0].max().item()
            threshold = min(0.3, cam_max * 0.5)
            for feature in ontology[pred_class]:
                if cam_max > threshold:
                    confidence = "strong" if cam_max > 0.7 else "moderate"
                    features_detected.append(f"{feature} ({confidence})")

            domain_name = list(domain_map.keys())[list(domain_map.values()).index(domain)]

            plt.figure(figsize=(24, 6))
            plt.subplot(1, 4, 1)
            plt.imshow(rgb_img)
            plt.title("Raw Image", fontsize=14)
            plt.axis('off')
            plt.subplot(1, 4, 2)
            plt.imshow(hires_vis)
            plt.title(f"HiResCAM\nPred: {pred_class}, True: {true_class}", fontsize=14)
            plt.axis('off')
            plt.colorbar()
            plt.subplot(1, 4, 3)
            plt.imshow(layer_vis)
            plt.title(f"LayerCAM\nDomain: {domain_name}", fontsize=14)
            plt.axis('off')
            plt.colorbar()
            plt.subplot(1, 4, 4)
            plt.imshow(diff_vis)
            plt.title("Domain Effect", fontsize=14)
            plt.axis('off')
            plt.colorbar()
            plt.suptitle(
                f"Sample {sample_count+1}\n"
                f"Features: {', '.join(features_detected) if features_detected else 'None'}\n"
                f"ROAD Score: {road_score_mean:.4f}",
                fontsize=16, y=1.1
            )
            plt.tight_layout()
            plt.show()

            print(f"Sample {sample_count+1}: Predicted={pred_class}, True={true_class}, ROAD Score={road_score_mean:.4f}, Domain={domain_name}")
            sample_count += 1
        if sample_count >= num_samples:
            break
