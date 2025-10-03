import os
import torch
from model import DomainAdaptiveViT  # Import from model.py
from losses import calculate_class_weights  # Import from losses.py
from dataset import create_data_loaders  # Import from dataset.py
from utils import plot_training_dynamics  # Import from utils.py

# Define NET_FOLDER for each model variant
MODEL_CONFIGS = {
    'dat_fixed': {
        'net_folder': 'skin_lesion_model_dat_fixed',
        'use_domain_adaptation': True,
        'fixed_embeddings': True
    },
    'dat_learned': {
        'net_folder': 'skin_lesion_model_dat_learned',
        'use_domain_adaptation': True,
        'fixed_embeddings': False
    },
    'vit_baseline': {
        'net_folder': 'skin_lesion_model_dat_vitbaseline',
        'use_domain_adaptation': False,
        'fixed_embeddings': False
    }
}

BEST_WEIGHTS_FILE = 'best_model'

# Domain map (define here or import)
domain_map = {
    'torso': 0, 'lower extremity': 1, 'upper extremity': 2, 'anterior torso': 3,
    'head/neck': 4, 'posterior torso': 5, 'palms/soles': 6, 'oral/genital': 7,
    'lateral torso': 8, 'unknown': 9
}

# Diagnosis to target map (define here or import)
diagnosis_to_target = {
    'AK': 0, 'BCC': 1, 'BKL': 2, 'DF': 3, 'MEL': 4, 'NV': 5, 'SCC': 6, 'UNK': 7, 'VASC': 8
}

# Load data loaders (only test_dataloader needed)
csv_path = "/kaggle/input/csvfile/whole_data_no_duplicates_kaggle.csv"
_, _, test_dataloader = create_data_loaders(csv_path)

# Class weights (for loss, if needed)
class_weights_tensor = calculate_class_weights(train_df, diagnosis_to_target)  # train_df from dataset.py or load here

def test_model(model_type):
    config = MODEL_CONFIGS.get(model_type)
    if not config:
        raise ValueError(f"Invalid model_type. Choose from: 'dat_fixed', 'dat_learned', 'vit_baseline'")

    net_folder = config['net_folder']
    best_checkpoint_path = os.path.join(net_folder, f"{BEST_WEIGHTS_FILE}.ckpt")
    if os.path.exists(best_checkpoint_path):
        print(f"Loading best checkpoint for {model_type}: {best_checkpoint_path}")
        model = DomainAdaptiveViT.load_from_checkpoint(best_checkpoint_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        # Test the model
        model.evaluator.evaluate(test_dataloader)

        # Save model's state dict
        torch.save(model.state_dict(), f"trained_model_{model_type}.pth")

        # Plot training dynamics (if losses collected during training)
        plot_training_dynamics(model.train_losses, model.val_losses, model.train_accuracies, model.val_accuracies)
    else:
        print(f"Best checkpoint not found at {best_checkpoint_path}. Cannot proceed with testing.")

# Example usage: Test all models
for model_type in ['dat_fixed', 'dat_learned', 'vit_baseline']:
    test_model(model_type)
