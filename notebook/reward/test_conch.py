import sys
import os
import torch

# Add project root to sys.path to access 'pfm'
# Assuming we are in notebook/reward/
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

print(f"Project path added: {project_root}")

try:
    print("Attempting to import pfm.conch.conch.factory...")
    from pfm.conch.conch.factory import create_model_from_pretrained
    from pfm.conch.conch.custom_tokenizer import get_tokenizer
    print("Import successful.")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

conch_path = "/data/ckpt/conch/pytorch_model.bin"
if not os.path.exists(conch_path):
    print(f"Error: Conch path {conch_path} does not exist.")
else:
    print(f"Conch path found: {conch_path}")

try:
    print("Loading Conch model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, transform = create_model_from_pretrained(
        model_cfg='conch_ViT-B-16',
        checkpoint_path=conch_path,
        device=device
    )
    model.eval()
    print("Conch model loaded successfully.")
except Exception as e:
    print(f"Failed to load Conch model: {e}")
    import traceback
    traceback.print_exc()
