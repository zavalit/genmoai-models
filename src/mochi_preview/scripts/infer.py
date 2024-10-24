import json
import os
import tempfile
import time

import click
import numpy as np
import torch
from PIL import Image

from mochi_preview.dit.joint_model.asymm_models_joint import FLASH_ATTN_IS_AVAILABLE
from mochi_preview.pipelines import (
    DecoderModelFactory,
    DitModelFactory,
    MochiMultiGPUPipeline,
    MochiSingleGPUPipeline,
    T5ModelFactory,
    linear_quadratic_schedule,
)
from mochi_preview.progress import progress_bar

pipeline = None
model_dir_path = None
num_gpus = torch.cuda.device_count()


def set_model_path(path):
    global model_dir_path
    model_dir_path = path


def load_model():
    global num_gpus, pipeline, model_dir_path
    if pipeline is None:
        MOCHI_DIR = model_dir_path
        print(f"Launching with {num_gpus} GPUs. If you want to force single GPU mode use CUDA_VISIBLE_DEVICES=0.")
        klass = MochiSingleGPUPipeline if num_gpus == 1 else MochiMultiGPUPipeline
        kwargs = dict(
            text_encoder_factory=T5ModelFactory(),
            dit_factory=DitModelFactory(model_path=f"{MOCHI_DIR}/dit.safetensors", model_dtype="bf16"),
            decoder_factory=DecoderModelFactory(
                model_path=f"{MOCHI_DIR}/vae.safetensors",
                model_stats_path=f"{MOCHI_DIR}/vae_stats.json",
            ),
        )
        if num_gpus > 1:
            kwargs["world_size"] = num_gpus
        pipeline = klass(**kwargs)


def generate_video(
    prompt,
    negative_prompt,
    width,
    height,
    num_frames,
    seed,
    cfg_scale,
    num_inference_steps,
):
    load_model()

    # sigma_schedule should be a list of floats of length (num_inference_steps + 1),
    # such that sigma_schedule[0] == 1.0 and sigma_schedule[-1] == 0.0 and monotonically decreasing.
    sigma_schedule = linear_quadratic_schedule(num_inference_steps, 0.025)

    # cfg_schedule should be a list of floats of length num_inference_steps.
    # For simplicity, we just use the same cfg scale at all timesteps,
    # but more optimal schedules may use varying cfg, e.g:
    # [5.0] * (num_inference_steps // 2) + [4.5] * (num_inference_steps // 2)
    cfg_schedule = [cfg_scale] * num_inference_steps

    args = {
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "sigma_schedule": sigma_schedule,
        "cfg_schedule": cfg_schedule,
        "num_inference_steps": num_inference_steps,
        # We *need* flash attention to batch cfg
        # and it's only worth doing in a high-memory regime (assume multiple GPUs)
        # "batch_cfg": FLASH_ATTN_IS_AVAILABLE and num_gpus > 1,
        "batch_cfg": False,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
    }

    final_frames = pipeline(**args)

    final_frames = final_frames[0]

    assert isinstance(final_frames, np.ndarray)
    assert final_frames.dtype == np.float32

    os.makedirs("outputs", exist_ok=True)
    output_path = os.path.join("outputs", f"output_{int(time.time())}.mp4")

    with tempfile.TemporaryDirectory() as tmpdir:
        frame_paths = []
        for i, frame in enumerate(final_frames):
            frame = (frame * 255).astype(np.uint8)
            frame_img = Image.fromarray(frame)
            frame_path = os.path.join(tmpdir, f"frame_{i:04d}.png")
            frame_img.save(frame_path)
            frame_paths.append(frame_path)

        frame_pattern = os.path.join(tmpdir, "frame_%04d.png")
        ffmpeg_cmd = f"ffmpeg -y -r 30 -i {frame_pattern} -vcodec libx264 -pix_fmt yuv420p {output_path}"
        os.system(ffmpeg_cmd)

        json_path = os.path.splitext(output_path)[0] + ".json"
        with open(json_path, "w") as f:
            json.dump(args, f, indent=4)

    return output_path


@click.command()
@click.option("--prompt", required=True, help="Prompt for video generation.")
@click.option("--negative_prompt", default="", help="Negative prompt for video generation.")
@click.option("--width", default=848, type=int, help="Width of the video.")
@click.option("--height", default=480, type=int, help="Height of the video.")
@click.option("--num_frames", default=163, type=int, help="Number of frames.")
@click.option("--seed", default=12345, type=int, help="Random seed.")
@click.option("--cfg_scale", default=4.5, type=float, help="CFG Scale.")
@click.option("--num_steps", default=64, type=int, help="Number of inference steps.")
@click.option("--model_dir", required=True, help="Path to the model directory.")
def generate_cli(
    prompt,
    negative_prompt,
    width,
    height,
    num_frames,
    seed,
    cfg_scale,
    num_steps,
    model_dir,
):
    set_model_path(model_dir)
    output = generate_video(
        prompt,
        negative_prompt,
        width,
        height,
        num_frames,
        seed,
        cfg_scale,
        num_steps,
    )
    click.echo(f"Video generated at: {output}")


if __name__ == "__main__":
    generate_cli()
