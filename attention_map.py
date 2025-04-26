import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms.functional import center_crop
import os
from models import get_model

def extract_attention_maps(model, x):
    """Extract attention maps from DINOv2 model."""
    attention_maps = []
    
    def hook_fn(module, input, output):
        # DINOv2 attention module doesn't directly expose attention maps
        # We need to compute them from the input and qkv weights
        q = module.q(input[0])  # Shape: [B, N, C]
        k = module.k(input[0])  # Shape: [B, N, C]
        
        # Compute attention scores (dot product of q and k)
        attn = (q @ k.transpose(-2, -1)) * module.scale  # Shape: [B, N, N]
        attn = attn.softmax(dim=-1)  # Apply softmax to get attention weights
        attention_maps.append(attn.detach())
    
    # Find blocks with attention modules
    hooks = []
    for name, module in model.base_model.model.named_modules():
        if 'attn' in name and hasattr(module, 'q') and hasattr(module, 'k'):
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)
    
    # Run forward pass
    with torch.no_grad():
        _ = model(x.unsqueeze(0))
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    if not attention_maps:
        # Fallback: Gradient-based attention visualization
        return None
    
    # Return the last attention map (from the final layer)
    return attention_maps[-1][0]  # Shape: [N, N]

def create_visualization(image_path, model, output_path='output.png'):
    # Load the image
    original_image = Image.open(image_path).convert('RGB')
    
    # Define transforms for center crop
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Get the center-cropped image (without resize)
    center_size = min(original_image.size)
    center_cropped_pil = center_crop(original_image, center_size)
    
    # Convert to tensor for model input (224x224)
    center_cropped_resized = center_cropped_pil.resize((224, 224), Image.BICUBIC)
    center_cropped_tensor = transform(center_cropped_resized)
    
    # Try to get attention maps - if it fails, use gradient-based method
    try:
        attn_map = extract_attention_maps(model, center_cropped_tensor)
        
        if attn_map is None:
            # Fallback: CAM-based approach
            print("Using CAM fallback for attention visualization")
            center_cropped_tensor.requires_grad_(True)
            model.zero_grad()
            outputs = model(center_cropped_tensor.unsqueeze(0), return_feature=True)
            
            # Handle tuple output
            if isinstance(outputs, tuple):
                feature = outputs[0]  # First element is likely the feature
            else:
                feature = outputs
                
            # Sum across feature dimensions
            feature_sum = feature.sum()
            feature_sum.backward()
            
            # Get gradients for CAM
            gradients = center_cropped_tensor.grad.abs()
            attn_map = gradients.sum(0).cpu().numpy()  # Sum across channels
        else:
            # Average over CLS token row and all other token rows
            attn_map = attn_map.mean(0).cpu().numpy()
            
            # For visualization, reshape to square grid if it's a sequence
            # (Remove CLS token if present)
            seq_len = attn_map.shape[0]
            if seq_len > 1:  # Remove CLS token if it exists
                grid_size = int(np.sqrt(seq_len - 1))
                if grid_size ** 2 == seq_len - 1:
                    attn_map = attn_map[1:].reshape(grid_size, grid_size)
                else:
                    # If not perfect square, just use as is
                    attn_map = attn_map[1:]
    except Exception as e:
        print(f"Error extracting attention maps: {e}")
        # Fallback method: Just use original image as placeholder
        attn_map = None
    
    # Now create patches and get scores for each patch
    patch_size = 224
    width, height = original_image.size
    
    # Calculate how many patches we can fit
    n_patches_w = width // patch_size
    n_patches_h = height // patch_size
    
    # Initialize score map
    score_map = np.zeros((n_patches_h, n_patches_w))
    
    for i in range(n_patches_h):
        for j in range(n_patches_w):
            # Extract patch
            left = j * patch_size
            top = i * patch_size
            right = left + patch_size
            bottom = top + patch_size
            
            patch = original_image.crop((left, top, right, bottom))
            patch_tensor = transform(patch)
            
            # Forward pass to get score
            with torch.no_grad():
                output = model(patch_tensor.unsqueeze(0))
                
                # Process the output to get a score
                if isinstance(output, tuple):
                    # If the model returns multiple outputs, take the first one
                    main_output = output[0]
                else:
                    main_output = output
                
                # Get a scalar score from the output tensor
                if isinstance(main_output, torch.Tensor):
                    # If it's a classification output
                    if main_output.dim() > 1 and main_output.size(1) > 1:
                        # Use max probability as score
                        score = torch.softmax(main_output, dim=1).max().item()
                    else:
                        # Single value output
                        score = torch.sigmoid(main_output).item()
                else:
                    # Fallback for unexpected types
                    score = 0.0
            
            score_map[i, j] = score
    
    # Create the visualization
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Original image
    axs[0].imshow(np.array(original_image))
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    
    # 2. Attention map overlay
    if attn_map is not None:
        # Normalize for visualization
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        
        # Resize attention map to match center crop size
        from scipy.ndimage import zoom
        
        # Calculate zoom factor
        zoom_factor = center_cropped_pil.size[0] / attn_map.shape[1]
        att_map_upscaled = zoom(attn_map, zoom_factor, order=1)
        
        # Create an overlay
        axs[1].imshow(np.array(center_cropped_pil))
        heatmap = axs[1].imshow(att_map_upscaled, alpha=0.5, cmap='jet')
        axs[1].set_title('Attention Map Overlay')
        axs[1].axis('off')
        plt.colorbar(heatmap, ax=axs[1], fraction=0.046, pad=0.04)
    else:
        axs[1].imshow(np.array(center_cropped_pil))
        axs[1].set_title('Center Crop (No Attention Map)')
        axs[1].axis('off')
    
    # 3. Patch scores heatmap
    im = axs[2].imshow(score_map, cmap='hot', interpolation='nearest')
    axs[2].set_title('Patch Scores Heatmap')
    axs[2].axis('off')
    plt.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

if __name__ == "__main__":
    # Path to your image
    image_path = "/root/autodl-tmp/AIGC_data/MSCOCO_XL/val2017/000000000139.png"
    
    # Get a DINOv2-LoRA model
    model_name = "DINOv2-LoRA:dinov2_vitl14"
    model = get_model(model_name, lora_rank=8, lora_alpha=1.0)
    state_dict = torch.load("/root/autodl-tmp/code/VAE_RESIZE_AIGC_detection/checkpoints/flux_double_resize/model_epoch_0.pth", map_location='cpu')['model']
    model.load_state_dict(state_dict)
    model.eval()
    
    # Create visualization
    output_path = create_visualization(image_path, model)
    print(f"Visualization saved to {output_path}")