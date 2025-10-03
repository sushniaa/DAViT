import torch
import torch.nn as nn
from transformers import ViTModel
from pytorch_lightning import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data.sampler import WeightedRandomSampler
from collections import defaultdict
import numpy as np

class DomainAdaptiveAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, domain_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.domain_dim = domain_dim
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.domain_scaling = nn.Linear(domain_dim, num_heads)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, domain_embedding):
        batch_size, seq_len, embed_dim = x.shape
        q = self.query(x).view(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 1, 3)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 1, 3)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 1, 3)
        domain_scaling = self.domain_scaling(domain_embedding).unsqueeze(-1).unsqueeze(-1)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.embed_dim / self.num_heads, dtype=torch.float32))
        attn_scores = attn_scores * domain_scaling
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)
        return self.proj(attn_output)

class DomainAdaptiveViT(LightningModule):
    def __init__(self, n_classes, domain_map, domain_embedding_dim=64,
                 use_domain_adaptation=True, fixed_embeddings=True, class_weights_tensor=None):
        super().__init__()
        self.save_hyperparameters()
        self.n_classes = n_classes
        self.domain_map = domain_map
        self.domain_embedding_dim = domain_embedding_dim
        self.use_domain_adaptation = use_domain_adaptation
        self.fixed_embeddings = fixed_embeddings
        self.class_weights_tensor = class_weights_tensor if class_weights_tensor is not None else torch.ones(n_classes)

        # ViT Backbone
        try:
            self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
            print(f"Successfully loaded pre-trained weights for ViT: {sum(p.numel() for p in self.vit.parameters())} parameters")
        except Exception as e:
            print(f"Failed to load pre-trained weights: {e}")
            self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k", ignore_mismatched_sizes=True)

        # Domain-specific components
        if self.use_domain_adaptation:
            self.domain_embedding = nn.Embedding(len(domain_map), domain_embedding_dim)
            if self.fixed_embeddings:
                self.domain_embedding.weight.requires_grad = False
            self.domain_adaptive_attention = DomainAdaptiveAttention(
                embed_dim=768, num_heads=8, domain_dim=domain_embedding_dim
            )

        # Dropout and Classification Head
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(768, n_classes)

        # Loss and Evaluator
        self.loss_fn = FocalLoss(alpha=None, gamma=3, reduction='none')

        # L2 regularization strength for domain embeddings
        self.l2_lambda = 1e-3

        # Tracking
        self.training_losses = {domain: [] for domain in self.domain_map.values()}
        self.validation_losses = {domain: [] for domain in self.domain_map.values()}
        self.domain_weights = defaultdict(lambda: 1.0, {domain: 1.0 for domain in domain_map.values()})
        self.train_losses, self.train_accuracies = [], []
        self.val_losses, self.val_accuracies = [], []
        self.val_balanced_accuracies = []
        self.epoch_domain_train_losses = {domain: [] for domain in self.domain_map.values()}
        self.epoch_domain_val_losses = {domain: [] for domain in self.domain_map.values()}
        self.test_preds, self.test_labels, self.test_logits = [], [], []
        self.val_preds, self.val_labels = [], []

    def forward(self, x, domains):
        vit_output = self.vit(x).last_hidden_state
        if self.use_domain_adaptation:
            domain_embeddings = self.domain_embedding(domains)
            vit_output = self.domain_adaptive_attention(vit_output, domain_embeddings)
        pooled_output = vit_output.mean(dim=1)
        pooled_output = self.dropout(pooled_output)
        return self.fc(pooled_output)

    def training_step(self, batch, batch_idx):
        images, labels, domains = batch
        logits = self(images, domains)
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("Logits contain NaN or inf values!")
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)

        loss_fn = FocalLoss(alpha=self.class_weights_tensor.to(images.device), gamma=3, reduction='none')
        per_sample_loss = loss_fn(logits, labels)
        per_sample_loss = torch.nan_to_num(per_sample_loss, nan=0.0, posinf=1e6, neginf=-1e6)

        # Record per-sample losses for each domain
        for idx in range(len(domains)):
            domain = domains[idx].item()
            self.training_losses[domain].append(per_sample_loss[idx].item())

        # Compute weighted loss
        if self.use_domain_adaptation:
            weighted_losses = []
            for idx in range(len(per_sample_loss)):
                domain = domains[idx].item()
                weighted_loss = per_sample_loss[idx] * self.domain_weights[domain]
                weighted_losses.append(weighted_loss)

            weighted_loss = torch.mean(torch.stack(weighted_losses))

            # Add L2 regularization for domain embeddings (only for DAT_Learned)
            if not self.fixed_embeddings:
                l2_reg = torch.tensor(0.0, device=self.device)
                for param in self.domain_embedding.parameters():
                    l2_reg += torch.norm(param, p=2)
                weighted_loss += self.l2_lambda * l2_reg
        else:
            weighted_loss = per_sample_loss.mean()

        preds = torch.argmax(logits, dim=1)
        train_acc = (preds == labels).float().mean().nan_to_num(0.0)
        self.log("train_loss", weighted_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", train_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": weighted_loss, "train_acc": train_acc}

    def validation_step(self, batch, batch_idx):
        images, labels, domains = batch
        logits = self(images, domains)
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("Validation logits contain NaN or inf values!")
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)

        loss_fn = FocalLoss(alpha=self.class_weights_tensor.to(images.device), gamma=3, reduction='none')
        loss = loss_fn(logits, labels)
        for domain in torch.unique(domains):
            domain_mask = (domains == domain)
            if domain_mask.sum() > 0:
                domain_loss = loss[domain_mask].mean()
                if torch.isnan(domain_loss):
                    domain_loss = torch.tensor(0.0, device=self.device)
                self.validation_losses[domain.item()].append(domain_loss.item())

        preds = torch.argmax(logits, dim=1)
        self.val_preds.extend(preds.cpu().numpy())
        self.val_labels.extend(labels.cpu().numpy())

        val_acc = (preds == labels).float().mean()
        self.log("val_loss", loss.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", val_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"val_loss": loss.mean(), "val_acc": val_acc}

    def on_train_epoch_end(self):
        train_loss = self.trainer.callback_metrics.get("train_loss")
        train_acc = self.trainer.callback_metrics.get("train_acc")
        if train_loss is not None:
            self.train_losses.append(train_loss.item())
        if train_acc is not None:
            self.train_accuracies.append(train_acc.item())

        # Compute and log per-domain training losses
        for domain, losses in self.training_losses.items():
            if len(losses) > 0:
                mean_loss = sum(losses) / len(losses)
                self.epoch_domain_train_losses[domain].append(mean_loss)
            else:
                self.epoch_domain_train_losses[domain].append(0.0)

        # Clear training losses for the next epoch
        if self.current_epoch < self.trainer.max_epochs - 1:
            for domain in self.training_losses:
                self.training_losses[domain] = []

        # Compute domain weights using softmax
        domain_means = {}
        for domain, losses in self.training_losses.items():
            domain_means[domain] = torch.mean(torch.tensor(losses, device=self.device)) if losses else torch.tensor(1.0, device=self.device)
        exp_values = {domain: torch.exp(mean_loss) for domain, mean_loss in domain_means.items()}
        exp_sum = sum(exp_values.values())
        self.domain_weights = {domain: exp_val / exp_sum for domain, exp_val in exp_values.items()}

    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics.get("val_loss")
        val_acc = self.trainer.callback_metrics.get("val_acc")
        if val_loss is not None:
            self.val_losses.append(val_loss.item())
        if val_acc is not None:
            self.val_accuracies.append(val_acc.item())

        # Compute balanced accuracy
        if len(self.val_preds) > 0 and len(self.val_labels) > 0:
            val_balanced_acc = balanced_accuracy_score(self.val_labels, self.val_preds)
            self.log("val_balanced_acc", val_balanced_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.val_balanced_accuracies.append(val_balanced_acc)

        # Compute validation domain losses
        for domain, losses in self.validation_losses.items():
            mean_loss = mean(losses) if losses else 0.0
            self.epoch_domain_val_losses[domain].append(mean_loss)

        # Clear validation losses for the next epoch
        if self.current_epoch < self.trainer.max_epochs - 1:
            for domain in self.validation_losses:
                self.validation_losses[domain] = []

        # Clear predictions and labels for balanced accuracy
        self.val_preds = []
        self.val_labels = []

    def configure_optimizers(self):
        base_lr = 1e-4
        decay_factor = 0.9
        optimizer_grouped_parameters = []

        # Classification Head
        optimizer_grouped_parameters.append({
            "params": self.fc.parameters(),
            "lr": base_lr,
            "initial_lr": base_lr,
            "name": "fc"
        })

        # ViT Encoder Layers
        num_layers = len(self.vit.encoder.layer)
        for i, layer in enumerate(self.vit.encoder.layer):
            layer_lr = base_lr * (decay_factor ** (num_layers - 1 - i))
            optimizer_grouped_parameters.append({
                "params": layer.parameters(),
                "lr": layer_lr,
                "initial_lr": base_lr,
                "name": f"vit.encoder.layer.{i}"
            })

        # Domain Adaptation Components
        if self.use_domain_adaptation:
            domain_attention_lr = base_lr * (decay_factor ** num_layers)
            optimizer_grouped_parameters.append({
                "params": self.domain_adaptive_attention.parameters(),
                "lr": domain_attention_lr,
                "initial_lr": domain_attention_lr,
                "name": "domain_adaptive_attention"
            })

            domain_embedding_lr = base_lr * (decay_factor ** (num_layers + 1))
            if not self.fixed_embeddings:
                optimizer_grouped_parameters.append({
                    "params": self.domain_embedding.parameters(),
                    "lr": domain_embedding_lr,
                    "initial_lr": domain_embedding_lr,
                    "name": "domain_embedding"
                })

        # ViT Embeddings
        vit_embeddings_lr = base_lr * (decay_factor ** (num_layers + 2))
        optimizer_grouped_parameters.append({
            "params": self.vit.embeddings.parameters(),
            "lr": vit_embeddings_lr,
            "initial_lr": vit_embeddings_lr,
            "name": "vit.embeddings"
        })

        optimizer = AdamW(optimizer_grouped_parameters, weight_decay=3e-2)

        max_lrs = [group['lr'] for group in optimizer_grouped_parameters]
        num_batches_per_epoch = len(self.trainer.train_dataloader) if self.trainer.train_dataloader else 2579
        total_steps = self.trainer.max_epochs * num_batches_per_epoch
        scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lrs,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1e4
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"}
        }

    def test_step(self, batch, batch_idx):
        return self.evaluator.test_step(self, batch)

    def on_test_epoch_end(self):
        self.evaluator.on_test_epoch_end()

    def on_save_checkpoint(self, checkpoint):
        checkpoint['train_losses'] = self.train_losses
        checkpoint['train_accuracies'] = self.train_accuracies
        checkpoint['val_losses'] = self.val_losses
        checkpoint['val_accuracies'] = self.val_accuracies
        checkpoint['val_balanced_accuracies'] = self.val_balanced_accuracies
        checkpoint['epoch_domain_train_losses'] = self.epoch_domain_train_losses
        checkpoint['epoch_domain_val_losses'] = self.epoch_domain_val_losses

    def on_load_checkpoint(self, checkpoint):
        self.train_losses = checkpoint.get('train_losses', [])
        self.train_accuracies = checkpoint.get('train_accuracies', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])
        self.val_balanced_accuracies = checkpoint.get('val_balanced_accuracies', [])
        self.epoch_domain_train_losses = checkpoint.get('epoch_domain_train_losses', {domain: [] for domain in self.domain_map.values()})
        self.epoch_domain_val_losses = checkpoint.get('epoch_domain_val_losses', {domain: [] for domain in self.domain_map.values()})
