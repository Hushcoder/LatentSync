# New superres file created
from omegaconf import OmegaConf
from pathlib import Path
from inference import main


def apply_super_resolution(image, model):
    """Applies GFPGAN or CodeFormer to upscale the image."""
    if model == 'GFPGAN':
        from gfpgan import GFPGAN
        sr_model = GFPGAN()
    elif model == 'CodeFormer':
        from codeformer import CodeFormer
        sr_model = CodeFormer()
    else:
        return image  # No super-resolution applied
    
    # Enhanced image 
    return sr_model.enhance(image)

def super_resolve(video_path, audio_path, output_path, guidance_scale, inference_steps, seed, superres_model):
    config_path = Path("configs/unet/second_stage.yaml")
    config = OmegaConf.load(config_path)

    # Update configuration for super-resolution
    config["run"].update({
        "guidance_scale": guidance_scale,
        "inference_steps": inference_steps,
    })

    # Calling the main inference function to get the processed frames
    processed_frames = main(config=config, args=[video_path, audio_path, output_path, seed])

    # Apply super-resolution to each frame
    super_resolved_frames = [apply_super_resolution(frame, superres_model) for frame in processed_frames]

    # Save or return the super-resolved frames as needed
    return super_resolved_frames


