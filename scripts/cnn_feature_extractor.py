# scripts/cnn_feature_extractor.py
import torch
import timm

def load_cnn_backbone(model_name: str, checkpoint_path: str, device: torch.device):
    """
    Loads a timm model as a feature extractor (embeddings) and loads weights.
    Uses num_classes=0 so model outputs embeddings.

    NOTE:
    - strict=False is intentional because your checkpoint may include a classifier head
      while num_classes=0 removes it. This is normal.
    """
    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=0  # outputs embeddings
    )

    state = torch.load(checkpoint_path, map_location=device)
    # Some checkpoints might be nested as {"state_dict": ...}
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state, strict=False)

    model.to(device)
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    return model
