import torch
import numpy as np
from collections import Counter
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, classification_report, auc)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class BaseEvaluator:
    def __init__(self, class_labels, device=None, checkpoint_path=None, domain_map=None):
        self.class_labels = class_labels
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        self.domain_map = domain_map or {
            'torso': 0, 'lower extremity': 1, 'upper extremity': 2, 'anterior torso': 3,
            'head/neck': 4, 'posterior torso': 5, 'palms/soles': 6, 'oral/genital': 7,
            'lateral torso': 8, 'unknown': 9
        }
        self.test_preds = []
        self.test_labels = []
        self.test_logits = []
        self.test_images = []
        self.test_losses = []
        self.model = None

    def load_model(self):
        if not self.checkpoint_path or not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found at {self.checkpoint_path}")

        self.model = DomainAdaptiveViT(
            n_classes=len(self.class_labels),
            domain_map=self.domain_map,
            domain_embedding_dim=64,
            use_domain_adaptation=True,
            fixed_embeddings=True,
            class_weights_tensor=class_weights_tensor
        ).to(self.device)

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get('state_dict', checkpoint)
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}

        try:
            self.model.load_state_dict(state_dict)
            print("Checkpoint loaded successfully!")
        except RuntimeError as e:
            print(f"Error loading state dict: {e}")
            self.model.load_state_dict(state_dict, strict=False)
            print("Checkpoint loaded with strict=False.")
        self.model.eval()

    def test_step(self, model, batch):
        images, labels, domains = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        domains = domains.to(self.device)

        with torch.no_grad():
            logits = model(images, domains)
            loss = F.cross_entropy(logits, labels)
            preds = torch.argmax(logits, dim=1)
            acc = (preds == labels).float().mean()

        self.test_preds.append(preds.cpu())
        self.test_labels.append(labels.cpu())
        self.test_logits.append(logits.cpu())
        self.test_images.extend(images.cpu())
        self.test_losses.append(loss.item())

        print(f"Batch {len(self.test_preds)}: Collected {len(preds)} predictions")
        return {"test_loss": loss.item(), "test_acc": acc.item()}

    def compute_specificity(self, labels, preds, n_classes):
        cm = confusion_matrix(labels, preds, labels=np.arange(n_classes))
        specificity = []
        for i in range(n_classes):
            tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
            fp = cm[:, i].sum() - cm[i, i]
            specificity.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
        return specificity

    def evaluate(self, test_dataloader):
        if self.model is None:
            self.load_model()
        for batch_idx, batch in enumerate(test_dataloader):
            self.test_step(self.model, batch)
        self.on_test_epoch_end()

    def on_test_epoch_end(self):
        if not self.test_preds:
            print("Warning: No predictions were collected during testing!")
            return

        test_preds = torch.cat(self.test_preds).numpy()
        test_labels = torch.cat(self.test_labels).numpy()
        test_logits = torch.cat(self.test_logits).numpy()
        test_probs = F.softmax(torch.tensor(test_logits), dim=1).numpy()

        prob_sums = test_probs.sum(axis=1, keepdims=True)
        test_probs = test_probs / prob_sums
        test_probs = np.clip(test_probs, 1e-7, 1 - 1e-7)

        accuracy = accuracy_score(test_labels, test_preds)
        balanced_accuracy = balanced_accuracy_score(test_labels, test_preds)
        sensitivity = recall_score(test_labels, test_preds, average='macro', zero_division=0)
        specificity = self.compute_specificity(test_labels, test_preds, len(self.class_labels))
        dice_coefficient = f1_score(test_labels, test_preds, average='macro', zero_division=0)
        avg_loss = np.mean(self.test_losses) if self.test_losses else None

        try:
            auc_score = roc_auc_score(test_labels, test_probs, multi_class='ovr')
            print(f"Overall AUC: {auc_score:.4f}")
        except ValueError as e:
            print(f"Error computing AUC: {e}")
            auc_score = np.nan

        auc_scores_class = []
        for i in range(len(self.class_labels)):
            binary_labels = (test_labels == i).astype(int)
            if len(np.unique(binary_labels)) < 2:
                auc_scores_class.append(np.nan)
            else:
                per_class_auc = roc_auc_score(binary_labels, test_probs[:, i])
                auc_scores_class.append(per_class_auc)

        print("\nTest Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
        print(f"Sensitivity: {sensitivity:.4f}")
        print(f"Specificity: {[f'{s:.4f}' for s in specificity]}")
        print(f"Dice Coefficient: {dice_coefficient:.4f}")
        print(f"AUC: {auc_score:.4f}")
        print(f"Per-class AUC scores: {[f'{a:.4f}' if not np.isnan(a) else 'NaN' for a in auc_scores_class]}")
        if avg_loss is not None:
            print(f"Test Loss: {avg_loss:.4f}")

        self.plot_roc_curve(test_labels, test_probs)
        self.plot_confusion_matrix(test_labels, test_preds)
        self.plot_predicted_images(test_labels, test_preds)

    def plot_roc_curve(self, labels, probs):
        n_classes = probs.shape[1]
        labels_binarized = label_binarize(labels, classes=np.arange(n_classes))
        plt.figure(figsize=(8, 6))
        for i in range(n_classes):
            if len(np.unique(labels_binarized[:, i])) > 1:
                fpr, tpr, _ = roc_curve(labels_binarized[:, i], probs[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f"{self.class_labels[i]} (AUC: {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], 'k--', label="Random Guessing")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Multi-Class ROC Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig("roc_curve.png")
        plt.show()

    def plot_confusion_matrix(self, labels, preds):
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=self.class_labels, yticklabels=self.class_labels)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.savefig("confusion_matrix.png")
        plt.show()

    def plot_predicted_images(self, labels, preds):
        print("\nPlotting first 10 images with predictions:")
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()
        for i in range(min(10, len(self.test_images))):
            img = self.test_images[i].permute(1, 2, 0).numpy()
            img = (img - img.min()) / (img.max() - img.min())
            axes[i].imshow(img)
            true_class = self.class_labels[labels[i]]
            pred_class = self.class_labels[preds[i]]
            axes[i].set_title(f"True: {true_class}\nPred: {pred_class}")
            axes[i].axis('off')
        plt.tight_layout()
        plt.savefig("predicted_images.png")
        plt.show()
