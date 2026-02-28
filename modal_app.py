"""
Modal Configuration for CMU 10799 Diffusion Homework

Defines Modal environment and training functions for cloud GPU training.

See docs/QUICKSTART-MODAL.md for setup and usage instructions.

All parameters are read from config YAML files first, then overridden by command-line arguments.
"""

import modal

# =============================================================================
# Modal App Definition
# =============================================================================

# Create the Modal app
app = modal.App("cmu-10799-diffusion")

# Define the container image with all dependencies
# This mirrors the CPU-only environment (environments/environment-cpu.yml)
# but installs GPU-enabled PyTorch automatically on Modal's GPU machines
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.21.0",
        "pillow>=9.0.0",
        "pyyaml>=6.0",
        "einops>=0.6.0",
        "tqdm>=4.64.0",
        "scipy>=1.9.0",
        "wandb>=0.15.0",
        "datasets>=2.0.0",
        "pyarrow<23.0.0",  # For HuggingFace Hub dataset loading
        "torch-fidelity>=0.3.0",  # Comprehensive evaluation metrics
    )
    # Copy the local project directory into the image
    .add_local_dir(".", "/root", ignore=["data",".git", ".venv*", "venv", "__pycache__", "logs", "checkpoints", "*.md", "docs", "environments", "notebooks"])
)

# Create a persistent volume for checkpoints and data
volume = modal.Volume.from_name("cmu-10799-diffusion-data", create_if_missing=True)

# =============================================================================
# Training Function
# =============================================================================

def _train_impl(
    method: str,
    config_path: str,
    resume_from: str,
    num_iterations: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    overfit_single_batch: bool = False,
):
    """
    Internal training implementation.

    Reads config from YAML file, applies command-line overrides.
    """
    import os
    import sys
    import yaml
    import tempfile
    import subprocess

    sys.path.insert(0, "/root")

    # Load config
    config_tag = method
    if config_path is None:
        config_path = f"/root/configs/{method}.yaml"
    else:
        config_path = f"/root/{config_path}"
        config_tag = os.path.splitext(os.path.basename(config_path))[0]

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Get num_gpus from config
    config_device = config['infrastructure'].get('device', 'cuda')
    num_gpus = config['infrastructure'].get('num_gpus', 1 if config_device == 'cuda' else 0)
    if num_gpus is None:
        num_gpus = 1 if config_device == 'cuda' else 0

    # Read from_hub from config
    from_hub = config['data'].get('from_hub', False)

    # Apply command-line overrides if provided
    if num_iterations is not None:
        config['training']['num_iterations'] = num_iterations
    if batch_size is not None:
        config['training']['batch_size'] = batch_size
    if learning_rate is not None:
        config['training']['learning_rate'] = learning_rate

    # Set Modal-specific paths
    config['data']['repo_name'] = "electronickale/cmu-10799-celeba64-subset"
    # Set root path for both modes:
    # - from_hub=true: checks for cached Arrow format first, then downloads from HF
    # - from_hub=false: looks for traditional folder structure (train/images/)
    config['data']['root'] = "/data/celeba"
    config['checkpoint']['dir'] = f"/data/checkpoints/{config_tag}"
    config['logging']['dir'] = f"/data/logs/{config_tag}"

    # Create directories
    os.makedirs(config['checkpoint']['dir'], exist_ok=True)
    os.makedirs(config['logging']['dir'], exist_ok=True)

    resume_path = f"/data/{resume_from}" if resume_from else None

    # Use torchrun for multi-GPU, direct import for single GPU
    if num_gpus > 1:
        temp_config_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as temp_file:
                yaml.safe_dump(config, temp_file)
                temp_config_path = temp_file.name

            cmd = [
                "torchrun",
                "--standalone",
                f"--nproc_per_node={num_gpus}",
                "/root/train.py",
                "--method", method,
                "--config", temp_config_path,
            ]
            if resume_path:
                cmd.extend(["--resume", resume_path])
            if overfit_single_batch:
                cmd.append("--overfit-single-batch")

            subprocess.run(cmd, check=True)
        finally:
            if temp_config_path and os.path.exists(temp_config_path):
                os.remove(temp_config_path)
    else:
        from train import train as run_training
        run_training(method_name=method, config=config, resume_path=resume_path, overfit_single_batch=overfit_single_batch)

    volume.commit()
    return f"Training complete! Checkpoints saved to /data/checkpoints/{method}"


# Create training functions for different GPU counts
@app.function(image=image, gpu="L40S:1", timeout=60*60*12, volumes={"/data": volume}, secrets=[modal.Secret.from_name("wandb-api-key")])
def train_1gpu(method: str = "ddpm", config_path: str = None, resume_from: str = None, num_iterations: int = None, batch_size: int = None, learning_rate: float = None, overfit_single_batch: bool = False):
    return _train_impl(method, config_path, resume_from, num_iterations, batch_size, learning_rate, overfit_single_batch)

@app.function(image=image, gpu="L40S:2", timeout=60*60*12, volumes={"/data": volume}, secrets=[modal.Secret.from_name("wandb-api-key")])
def train_2gpu(method: str = "ddpm", config_path: str = None, resume_from: str = None, num_iterations: int = None, batch_size: int = None, learning_rate: float = None, overfit_single_batch: bool = False):
    return _train_impl(method, config_path, resume_from, num_iterations, batch_size, learning_rate, overfit_single_batch)

@app.function(image=image, gpu="L40S:3", timeout=60*60*12, volumes={"/data": volume}, secrets=[modal.Secret.from_name("wandb-api-key")])
def train_3gpu(method: str = "ddpm", config_path: str = None, resume_from: str = None, num_iterations: int = None, batch_size: int = None, learning_rate: float = None, overfit_single_batch: bool = False):
    return _train_impl(method, config_path, resume_from, num_iterations, batch_size, learning_rate, overfit_single_batch)

@app.function(image=image, gpu="L40S:4", timeout=60*60*12, volumes={"/data": volume}, secrets=[modal.Secret.from_name("wandb-api-key")])
def train_4gpu(method: str = "ddpm", config_path: str = None, resume_from: str = None, num_iterations: int = None, batch_size: int = None, learning_rate: float = None, overfit_single_batch: bool = False):
    return _train_impl(method, config_path, resume_from, num_iterations, batch_size, learning_rate, overfit_single_batch)

@app.function(image=image, gpu="L40S:5", timeout=60*60*12, volumes={"/data": volume}, secrets=[modal.Secret.from_name("wandb-api-key")])
def train_5gpu(method: str = "ddpm", config_path: str = None, resume_from: str = None, num_iterations: int = None, batch_size: int = None, learning_rate: float = None, overfit_single_batch: bool = False):
    return _train_impl(method, config_path, resume_from, num_iterations, batch_size, learning_rate, overfit_single_batch)

@app.function(image=image, gpu="L40S:6", timeout=60*60*12, volumes={"/data": volume}, secrets=[modal.Secret.from_name("wandb-api-key")])
def train_6gpu(method: str = "ddpm", config_path: str = None, resume_from: str = None, num_iterations: int = None, batch_size: int = None, learning_rate: float = None, overfit_single_batch: bool = False):
    return _train_impl(method, config_path, resume_from, num_iterations, batch_size, learning_rate, overfit_single_batch)

@app.function(image=image, gpu="L40S:7", timeout=60*60*12, volumes={"/data": volume}, secrets=[modal.Secret.from_name("wandb-api-key")])
def train_7gpu(method: str = "ddpm", config_path: str = None, resume_from: str = None, num_iterations: int = None, batch_size: int = None, learning_rate: float = None, overfit_single_batch: bool = False):
    return _train_impl(method, config_path, resume_from, num_iterations, batch_size, learning_rate, overfit_single_batch)

@app.function(image=image, gpu="L40S:8", timeout=60*60*12, volumes={"/data": volume}, secrets=[modal.Secret.from_name("wandb-api-key")])
def train_8gpu(method: str = "ddpm", config_path: str = None, resume_from: str = None, num_iterations: int = None, batch_size: int = None, learning_rate: float = None, overfit_single_batch: bool = False):
    return _train_impl(method, config_path, resume_from, num_iterations, batch_size, learning_rate, overfit_single_batch)

# Map GPU counts to functions
TRAIN_FUNCTIONS = {
    1: train_1gpu,
    2: train_2gpu,
    3: train_3gpu,
    4: train_4gpu,
    5: train_5gpu,
    6: train_6gpu,
    7: train_7gpu,
    8: train_8gpu,
}


# =============================================================================
# Progressive Distillation Function
# =============================================================================

@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60 * 24,  # 24 hours (5 stages x ~4h each)
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("wandb-api-key")],
)
def distill(
    config_path: str = "configs/pd_distill.yaml",
    teacher_checkpoint: str = None,
    resume_stage: int = None,
    resume_checkpoint: str = None,
):
    """Run Progressive Distillation training on Modal."""
    import os
    import sys
    import yaml
    import tempfile
    import subprocess

    sys.path.insert(0, "/root")

    with open(f"/root/{config_path}", "r") as f:
        config = yaml.safe_load(f)

    config["data"]["root"] = "/data/celeba"
    config["data"]["repo_name"] = "electronickale/cmu-10799-celeba64-subset"
    config["logging"]["dir"] = "/data/logs/pd_distill"

    if teacher_checkpoint is not None:
        config["distillation"]["teacher_checkpoint"] = f"/data/{teacher_checkpoint}"
    elif not config["distillation"]["teacher_checkpoint"].startswith("/data"):
        config["distillation"]["teacher_checkpoint"] = (
            f"/data/{config['distillation']['teacher_checkpoint']}"
        )

    os.makedirs(config["logging"]["dir"], exist_ok=True)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tf:
        yaml.safe_dump(config, tf)
        tmp_cfg = tf.name

    cmd = ["python", "/root/train_distill.py", "--config", tmp_cfg]
    if resume_stage is not None:
        cmd.extend(["--resume_stage", str(resume_stage)])
    if resume_checkpoint is not None:
        cmd.extend(["--resume_checkpoint", f"/data/{resume_checkpoint}"])

    subprocess.run(cmd, check=True)
    volume.commit()

    os.remove(tmp_cfg)
    return "Progressive Distillation complete!"


# =============================================================================
# Sampling Function
# =============================================================================

@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60 * 3,  # 3 hours
    volumes={"/data": volume},
)
def sample(
    method: str = "ddpm",
    checkpoint: str = "checkpoints/ddpm/ddpm_final.pt",
    num_samples: int = None,
    num_steps: int = None,
    k: int = None,
    distilled_checkpoint: str = None,
    dws_tau: float = None,
    n_rx_steps: int = None,
    extrapolation: str = None,
):
    """
    Generate samples from a trained model.

    Uses sample.py via subprocess, similar to how training uses train.py.
    """
    import os
    import subprocess
    from datetime import datetime

    # Set up paths
    checkpoint_path = f"/data/{checkpoint}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"/data/samples/{method}_{timestamp}.png"

    os.makedirs("/data/samples", exist_ok=True)

    # Build command to run sample.py
    cmd = [
        "python", "/root/sample.py",
        "--checkpoint", checkpoint_path,
        "--method", method,
        "--grid",
        "--output", output_path,
    ]

    if num_samples is not None:
        cmd.extend(["--num_samples", str(num_samples)])
    if num_steps is not None:
        cmd.extend(["--num_steps", str(num_steps)])
    if k is not None:
        cmd.extend(["--k", str(k)])
    if distilled_checkpoint is not None:
        cmd.extend(["--distilled_checkpoint", f"/data/{distilled_checkpoint}"])
    if dws_tau is not None:
        cmd.extend(["--dws_tau", str(dws_tau)])
    if n_rx_steps is not None:
        cmd.extend(["--n_rx_steps", str(n_rx_steps)])
    if extrapolation is not None:
        cmd.extend(["--extrapolation", extrapolation])

    subprocess.run(cmd, check=True)
    volume.commit()

    return f"Samples saved to {output_path}"


# =============================================================================
# Dataset Download Function
# =============================================================================

@app.function(
    image=image,
    timeout=60 * 60,  # 1 hour
    volumes={"/data": volume},
)
def download_dataset():
    """
    Download the dataset from HuggingFace Hub to Modal volume.

    Caches the dataset in Arrow format at /data/celeba. After downloading,
    training with from_hub=true will automatically use this cached version
    instead of redownloading.
    """
    import sys
    sys.path.insert(0, "/root")

    from datasets import load_dataset
    import os

    print("Downloading dataset from HuggingFace Hub...")
    dataset = load_dataset("electronickale/cmu-10799-celeba64-subset")

    # Save to volume in Arrow format
    os.makedirs("/data/celeba", exist_ok=True)
    dataset.save_to_disk("/data/celeba")

    volume.commit()

    print(f"Dataset cached to /data/celeba")
    print(f"Train set size: {len(dataset['train'])}")
    return "Dataset download complete! Training with from_hub=true with root = '/data/celeba' will now use this cached version."


# =============================================================================
# Evaluation Function (using torch-fidelity)
# =============================================================================

@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60 * 8,  # 8 hours
    volumes={"/data": volume},
)
def evaluate_torch_fidelity(
    method: str = "ddpm",
    checkpoint: str = "checkpoints/ddpm/ddpm_final.pt",
    metrics: str = "fid,kid",
    num_samples: int = 5000,
    batch_size: int = 128,
    num_steps: int = None,
    k: int = None,
    override: bool = False,
    distilled_checkpoint: str = None,
    dws_tau: float = None,
    n_rx_steps: int = None,
    extrapolation: str = None,
):
    """
    Evaluate using torch-fidelity CLI.

    Uses the fidelity command to compute metrics directly.

    Args:
        method: Sampling method (ddpm, ddim, rx_ddim, dws_rx_ddim, etc.)
        checkpoint: Path to checkpoint (relative to /data)
        metrics: Comma-separated: 'fid', 'kid', 'is' (default: 'fid,kid')
        num_samples: Number of samples to generate
        batch_size: Batch size
        num_steps: Sampling steps (optional)
        override: Force regenerate samples even if they exist
        distilled_checkpoint: DWS-RX-DDIM: path to PD model (relative to /data)
        dws_tau: DWS-RX-DDIM: re-noising level
        n_rx_steps: DWS-RX-DDIM: number of RX refinement steps
        extrapolation: DWS-RX-DDIM: extrapolation strategy (standard, cng, do)
    """
    import sys
    import subprocess
    from pathlib import Path
    sys.path.insert(0, "/root")

    checkpoint_path = f"/data/{checkpoint}"

    # Put samples in same parent dir as checkpoint under samples/
    checkpoint_dir = Path(checkpoint_path).parent
    generated_dir = str(checkpoint_dir / "samples" / "generated")
    cache_dir = str(checkpoint_dir / "samples" / "cache")

    # Prepare dataset path for torch-fidelity
    # torch-fidelity needs actual image files, not Arrow format
    dataset_arrow_path = "/data/celeba"
    dataset_images_path = "/data/celeba_images"

    # Extract images from Arrow format if not already done
    import os
    if not os.path.exists(dataset_images_path):
        print("=" * 60)
        print("Extracting dataset images for torch-fidelity...")
        print("=" * 60)

        from datasets import load_from_disk

        dataset = load_from_disk(dataset_arrow_path)
        train_data = dataset['train']

        os.makedirs(dataset_images_path, exist_ok=True)

        print(f"Extracting {len(train_data)} images...")
        for idx, item in enumerate(train_data):
            img = item['image']
            img_path = os.path.join(dataset_images_path, f"{idx:06d}.png")
            img.save(img_path)

            if (idx + 1) % 1000 == 0:
                print(f"  Extracted {idx + 1}/{len(train_data)} images")

        volume.commit()
        print(f"Dataset images saved to {dataset_images_path}")
    else:
        print(f"Using cached dataset images at {dataset_images_path}")

    dataset_path = dataset_images_path

    # Step 1: Generate samples
    print("=" * 60)
    print("Step 1/2: Generating samples...")
    print("=" * 60)

    import os
    import shutil
    import glob

    # Check if samples already exist
    need_generation = True
    if os.path.exists(generated_dir) and not override:
        # Check for both png and jpg files
        existing_samples = (
            glob.glob(os.path.join(generated_dir, "*.png")) +
            glob.glob(os.path.join(generated_dir, "*.jpg")) + 
            glob.glob(os.path.join(generated_dir, "*.jpeg"))
        )
        num_existing = len(existing_samples)

        if num_existing >= num_samples:
            print(f"Found {num_existing} existing samples (need {num_samples})")
            print("Skipping sample generation (use --override to force regeneration)")
            need_generation = False
        else:
            print(f"Found {num_existing} existing samples but need {num_samples}")
            print("Regenerating samples...")
            shutil.rmtree(generated_dir)
    elif os.path.exists(generated_dir) and override:
        print("Override flag set, regenerating samples...")
        shutil.rmtree(generated_dir)

    if need_generation:
        sample_cmd = [
            "python", "/root/sample.py",
            "--checkpoint", checkpoint_path,
            "--method", method,
            "--output_dir", generated_dir,
            "--num_samples", str(num_samples),
            "--batch_size", str(batch_size),
        ]

        if num_steps:
            sample_cmd.extend(["--num_steps", str(num_steps)])
        if k is not None:
            sample_cmd.extend(["--k", str(k)])
        if distilled_checkpoint is not None:
            sample_cmd.extend(["--distilled_checkpoint", f"/data/{distilled_checkpoint}"])
        if dws_tau is not None:
            sample_cmd.extend(["--dws_tau", str(dws_tau)])
        if n_rx_steps is not None:
            sample_cmd.extend(["--n_rx_steps", str(n_rx_steps)])
        if extrapolation is not None:
            sample_cmd.extend(["--extrapolation", extrapolation])

        subprocess.run(sample_cmd, check=True)
        print(f"Generated {num_samples} samples to {generated_dir}")
    else:
        print(f"Using existing samples from {generated_dir}")

    # Step 2: Run fidelity
    print("\n" + "=" * 60)
    print("Step 2/2: Running torch-fidelity...")
    print("=" * 60)

    os.makedirs(cache_dir, exist_ok=True)

    fidelity_cmd = [
        "fidelity",
        "--gpu", "0",
        "--batch-size", str(batch_size),
        "--cache-root", cache_dir,
        "--input1", generated_dir,
        "--input2", dataset_path,
    ]

    if "fid" in metrics:
        fidelity_cmd.append("--fid")
    if "kid" in metrics:
        fidelity_cmd.append("--kid")
    if "is" in metrics or "isc" in metrics:
        fidelity_cmd.append("--isc")

    print(f"\nRunning command: {' '.join(fidelity_cmd)}\n")

    try:
        result = subprocess.run(fidelity_cmd, check=True, capture_output=True, text=True)
        volume.commit()
        return result.stdout
    except subprocess.CalledProcessError as e:
        # Print the error output to help debug
        print(f"\nError running fidelity command:")
        print(f"Command: {' '.join(fidelity_cmd)}")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print(f"\nStdout:\n{e.stdout}")
        if e.stderr:
            print(f"\nStderr:\n{e.stderr}")
        raise


# =============================================================================
# Benchmark Functions (RX-DDIM vs DDIM evaluation)
# =============================================================================

def _ensure_dataset_images():
    """Extract dataset images from Arrow format if not already cached."""
    import os
    dataset_arrow_path = "/data/celeba"
    dataset_images_path = "/data/celeba_images"

    if not os.path.exists(dataset_images_path):
        print("Extracting dataset images for torch-fidelity...")
        from datasets import load_from_disk

        dataset = load_from_disk(dataset_arrow_path)
        train_data = dataset["train"]
        os.makedirs(dataset_images_path, exist_ok=True)

        for idx, item in enumerate(train_data):
            item["image"].save(os.path.join(dataset_images_path, f"{idx:06d}.png"))
            if (idx + 1) % 1000 == 0:
                print(f"  Extracted {idx + 1}/{len(train_data)} images")

        volume.commit()
    return dataset_images_path


@app.function(image=image, gpu="L40S", timeout=60 * 60 * 2, volumes={"/data": volume})
def benchmark_kid_single(
    method: str,
    num_steps: int,
    checkpoint: str,
    num_samples: int = 5000,
    batch_size: int = 32,
    k: int = 2,
) -> str:
    """
    KID worker: generate samples for one (method, step_count) config and compute KID.

    Returns a CSV result line: "method,num_steps,nfe,kid_mean,kid_std"
    """
    import subprocess

    dataset_path = _ensure_dataset_images()
    checkpoint_path = f"/data/{checkpoint}"
    output_dir = f"/data/benchmark/{method}_{num_steps}"

    cmd = [
        "python", "/root/evaluate_rx_ddim.py",
        "--mode", "kid",
        "--checkpoint", checkpoint_path,
        "--method", method,
        "--num_steps", str(num_steps),
        "--num_samples", str(num_samples),
        "--batch_size", str(batch_size),
        "--k", str(k),
        "--output_dir", output_dir,
        "--dataset_path", dataset_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    result.check_returncode()

    # Parse the RESULT: line from stdout
    for line in result.stdout.strip().splitlines():
        if line.startswith("RESULT:"):
            volume.commit()
            return line[len("RESULT:"):]

    volume.commit()
    raise RuntimeError(f"No RESULT line found in output:\n{result.stdout}")


@app.function(image=image, gpu="L40S", timeout=60 * 60 * 6, volumes={"/data": volume})
def evaluate_pd_sweep(
    checkpoint: str,
    step_counts: str = "1,2,3,4,5,10",
    num_samples: int = 5000,
    batch_size: int = 32,
) -> str:
    """Run the full PD KID sweep across step_counts on one GPU.

    checkpoint: path relative to /data/ (e.g.
        logs/pd_distill/pd_distill_20260225_235450/checkpoints/pd_stage5_final.pt)

    Returns the path to the output CSV on the volume.
    """
    import subprocess

    dataset_path = _ensure_dataset_images()
    checkpoint_path = f"/data/{checkpoint}"
    output_csv = "/data/benchmark/kid_pd_vs_nfe.csv"

    cmd = [
        "python", "/root/evaluate_pd.py",
        "--mode", "sweep",
        "--checkpoint", checkpoint_path,
        "--step_counts", step_counts,
        "--num_samples", str(num_samples),
        "--batch_size", str(batch_size),
        "--dataset_path", dataset_path,
        "--output_csv", output_csv,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    result.check_returncode()

    volume.commit()
    return output_csv


@app.function(image=image, gpu="L40S", timeout=60 * 60 * 3, volumes={"/data": volume})
def evaluate_pd_fid(
    checkpoint: str,
    num_steps: int = 1,
    num_samples: int = 5000,
    batch_size: int = 32,
) -> str:
    """Run FID evaluation for PD v-prediction model at given step count."""
    import subprocess

    dataset_path = _ensure_dataset_images()
    checkpoint_path = f"/data/{checkpoint}"
    output_dir = f"/data/benchmark/pd_fid_{num_steps}steps"

    cmd = [
        "python", "/root/evaluate_pd.py",
        "--mode", "fid",
        "--checkpoint", checkpoint_path,
        "--num_steps", str(num_steps),
        "--num_samples", str(num_samples),
        "--batch_size", str(batch_size),
        "--dataset_path", dataset_path,
        "--output_dir", output_dir,
    ]

    subprocess.run(cmd, check=True)
    volume.commit()
    return f"FID complete (steps={num_steps}). Samples at {output_dir}"


@app.function(image=image, gpu="L40S", timeout=60 * 60 * 8, volumes={"/data": volume})
def evaluate_dws_sweep(
    base_checkpoint: str,
    distilled_checkpoint: str,
    rx_step_counts: str = "1,2,3,4,5,10",
    tau: float = 0.3,
    k: int = 2,
    extrapolation: str = "standard",
    num_samples: int = 5000,
    batch_size: int = 32,
) -> str:
    """Run DWS-RX-DDIM KID sweep across n_rx_steps on one GPU."""
    import subprocess

    dataset_path = _ensure_dataset_images()
    output_csv = "/data/benchmark/kid_dws_vs_nfe.csv"

    cmd = [
        "python", "/root/evaluate_dws.py",
        "--base_checkpoint", f"/data/{base_checkpoint}",
        "--distilled_checkpoint", f"/data/{distilled_checkpoint}",
        "--rx_step_counts", rx_step_counts,
        "--tau", str(tau),
        "--k", str(k),
        "--extrapolation", extrapolation,
        "--num_samples", str(num_samples),
        "--batch_size", str(batch_size),
        "--dataset_path", dataset_path,
        "--output_csv", output_csv,
    ]

    subprocess.run(cmd, check=True)
    volume.commit()
    return output_csv


@app.function(image=image, gpu="L40S", timeout=60 * 60 * 8, volumes={"/data": volume})
def evaluate_tau_sweep(
    base_checkpoint: str,
    distilled_checkpoint: str,
    tau_values: str = "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
    n_rx_steps: int = 1,
    k: int = 2,
    extrapolation: str = "standard",
    num_samples: int = 5000,
    batch_size: int = 32,
) -> str:
    """Run DWS-RX-DDIM KID sweep across tau values on one GPU."""
    import subprocess

    dataset_path = _ensure_dataset_images()
    output_csv = "/data/benchmark/kid_dws_tau_sweep.csv"

    cmd = [
        "python", "/root/evaluate_tau_sweep.py",
        "--base_checkpoint", f"/data/{base_checkpoint}",
        "--distilled_checkpoint", f"/data/{distilled_checkpoint}",
        "--tau_values", tau_values,
        "--n_rx_steps", str(n_rx_steps),
        "--k", str(k),
        "--extrapolation", extrapolation,
        "--num_samples", str(num_samples),
        "--batch_size", str(batch_size),
        "--dataset_path", dataset_path,
        "--output_csv", output_csv,
    ]

    subprocess.run(cmd, check=True)
    volume.commit()
    return output_csv


@app.function(image=image, gpu="L40S", timeout=60 * 60 * 10, volumes={"/data": volume})
def evaluate_dws_grid(
    base_checkpoint: str,
    distilled_checkpoint: str,
    tau_values: str = "0.05,0.1,0.2,0.3,0.4",
    rx_step_counts: str = "1,2,3,4,5,10",
    k: int = 2,
    extrapolation: str = "standard",
    num_samples: int = 5000,
    batch_size: int = 32,
) -> str:
    """Run DWS-RX-DDIM 2D grid sweep (tau x n_rx_steps) on one GPU."""
    import subprocess

    dataset_path = _ensure_dataset_images()
    output_csv = "/data/benchmark/kid_dws_grid.csv"

    cmd = [
        "python", "/root/evaluate_dws_grid.py",
        "--base_checkpoint", f"/data/{base_checkpoint}",
        "--distilled_checkpoint", f"/data/{distilled_checkpoint}",
        "--tau_values", tau_values,
        "--rx_step_counts", rx_step_counts,
        "--k", str(k),
        "--extrapolation", extrapolation,
        "--num_samples", str(num_samples),
        "--batch_size", str(batch_size),
        "--dataset_path", dataset_path,
        "--output_csv", output_csv,
    ]

    subprocess.run(cmd, check=True)
    volume.commit()
    return output_csv


@app.function(image=image, timeout=60, volumes={"/data": volume})
def write_kid_csv(kid_rows: list) -> str:
    """Write collected KID result lines to a CSV on the volume."""
    import os
    output_csv = "/data/benchmark/kid_vs_nfe.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w") as f:
        f.write("method,num_steps,nfe,kid_mean,kid_std\n")
        for row in kid_rows:
            f.write(row + "\n")
    volume.commit()
    return output_csv


@app.function(image=image, gpu="L40S", timeout=60 * 60, volumes={"/data": volume})
def benchmark_timing(
    checkpoint: str,
    step_counts: str = "1,3,5,10,25,50,100",
    batch_size: int = 32,
    num_warmup: int = 2,
    num_trials: int = 5,
    k: int = 2,
) -> str:
    """Run wall-clock timing benchmark for DDIM and RX-DDIM."""
    import subprocess

    checkpoint_path = f"/data/{checkpoint}"
    output_csv = "/data/benchmark/timing_benchmark.csv"

    cmd = [
        "python", "/root/benchmark_timing.py",
        "--checkpoint", checkpoint_path,
        "--step_counts", step_counts,
        "--batch_size", str(batch_size),
        "--num_warmup", str(num_warmup),
        "--num_trials", str(num_trials),
        "--k", str(k),
        "--output_csv", output_csv,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    result.check_returncode()

    volume.commit()
    return output_csv


@app.function(image=image, gpu="L40S", timeout=60 * 60, volumes={"/data": volume})
def benchmark_l2(
    checkpoint: str,
    num_samples_l2: int = 8,
    k: int = 2,
    seed: int = 42,
) -> str:
    """
    L2 worker: run full L2 error sweep on one GPU, write CSV to volume.

    Returns the path to the output CSV.
    """
    import subprocess

    checkpoint_path = f"/data/{checkpoint}"
    output_csv = "/data/benchmark/l2_vs_nfe.csv"

    cmd = [
        "python", "/root/evaluate_rx_ddim.py",
        "--mode", "l2",
        "--checkpoint", checkpoint_path,
        "--rx_step_counts", "1,5,10,25,50",
        "--num_samples_l2", str(num_samples_l2),
        "--k", str(k),
        "--seed", str(seed),
        "--output_csv", output_csv,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    result.check_returncode()

    volume.commit()
    return output_csv


# =============================================================================
# CLI Entry Points
# =============================================================================

@app.local_entrypoint()
def main(
    action: str = "train",
    method: str = "ddpm",
    config: str = None,
    checkpoint: str = None,
    resume: str = None,  # Path to checkpoint to resume training from (relative to /data/)
    resume_stage: int = None,  # For distill: 0-based stage index to resume from
    resume_checkpoint: str = None,  # For distill: mid-stage checkpoint to resume from
    iterations: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    num_samples: int = None,
    num_steps: int = None,
    k: int = None,
    metrics: str = None,
    overfit_single_batch: bool = False,
    override: bool = False,
    seed: int = None,
    distilled_checkpoint: str = None,
    dws_tau: float = None,
    n_rx_steps: int = None,
    extrapolation: str = None,
):
    """
    Main entry point for Modal CLI.

    See docs/QUICKSTART-MODAL.md for usage instructions.

    All parameters are read from config YAML files first, then overridden by command-line arguments.
    """
    if action == "download":
        result = download_dataset.remote()
        print(result)
    elif action == "train":
        # Read config to determine GPU count
        import yaml

        local_config_path = config or f"configs/{method}.yaml"
        with open(local_config_path, 'r') as f:
            local_config = yaml.safe_load(f)

        # Get num_gpus from config
        config_device = local_config['infrastructure'].get('device', 'cuda')
        num_gpus = local_config['infrastructure'].get('num_gpus', 1 if config_device == 'cuda' else 0)
        if num_gpus is None:
            num_gpus = 1 if config_device == 'cuda' else 0

        # Get the appropriate training function
        train_fn = TRAIN_FUNCTIONS.get(num_gpus)
        if train_fn is None:
            raise ValueError(
                f"Unsupported num_gpus={num_gpus} in config. "
                f"Supported: 1-8"
            )

        result = train_fn.remote(
            method=method,
            config_path=config,
            resume_from=resume,
            num_iterations=iterations,
            batch_size=batch_size,
            learning_rate=learning_rate,
            overfit_single_batch=overfit_single_batch,
        )
        print(result)
    elif action == "distill":
        dist_kwargs = {"config_path": config or "configs/pd_distill.yaml"}
        if checkpoint is not None:
            dist_kwargs["teacher_checkpoint"] = checkpoint
        if resume_stage is not None:
            dist_kwargs["resume_stage"] = resume_stage
        if resume_checkpoint is not None:
            dist_kwargs["resume_checkpoint"] = resume_checkpoint
        result = distill.remote(**dist_kwargs)
        print(result)
    elif action == "sample":
        if checkpoint is None:
            checkpoint = f"checkpoints/{method}/{method}_final.pt"
        result = sample.remote(
            method=method,
            checkpoint=checkpoint,
            num_samples=num_samples,
            num_steps=num_steps,
            k=k,
            distilled_checkpoint=distilled_checkpoint,
            dws_tau=dws_tau,
            n_rx_steps=n_rx_steps,
            extrapolation=extrapolation,
        )
        print(result)
    elif action == "evaluate" or action == "evaluate_torch_fidelity":
        if checkpoint is None:
            checkpoint = f"checkpoints/{method}/{method}_final.pt"

        eval_kwargs = {
            'method': method,
            'checkpoint': checkpoint,
            'override': override,
        }
        if metrics is not None:
            eval_kwargs['metrics'] = metrics
        if num_samples is not None:
            eval_kwargs['num_samples'] = num_samples
        if batch_size is not None:
            eval_kwargs['batch_size'] = batch_size
        if num_steps is not None:
            eval_kwargs['num_steps'] = num_steps
        if k is not None:
            eval_kwargs['k'] = k
        if distilled_checkpoint is not None:
            eval_kwargs['distilled_checkpoint'] = distilled_checkpoint
        if dws_tau is not None:
            eval_kwargs['dws_tau'] = dws_tau
        if n_rx_steps is not None:
            eval_kwargs['n_rx_steps'] = n_rx_steps
        if extrapolation is not None:
            eval_kwargs['extrapolation'] = extrapolation

        result = evaluate_torch_fidelity.remote(**eval_kwargs)
        print(result)
    elif action == "benchmark_timing":
        if checkpoint is None:
            checkpoint = "logs/ddpm_modal/ddpm_20260127_162352/checkpoints/ddpm_final.pt"

        timing_k = k if k is not None else 2
        timing_batch_size = batch_size if batch_size is not None else 32

        print(f"Launching timing benchmark on Modal (L40S)...")
        result_path = benchmark_timing.remote(
            checkpoint=checkpoint,
            batch_size=timing_batch_size,
            k=timing_k,
        )
        print(f"\nTiming results written to: {result_path}")
        print("Download with:")
        print("  modal volume get cmu-10799-diffusion-data benchmark/timing_benchmark.csv .")

    elif action == "benchmark":
        if checkpoint is None:
            checkpoint = "checkpoints/ddpm_modal/ddpm_final.pt"

        kid_num_samples = num_samples if num_samples is not None else 5000
        kid_batch_size = batch_size if batch_size is not None else 32
        kid_k = k if k is not None else 2
        benchmark_seed = seed if seed is not None else 42

        # RX-DDIM steps and their DDIM 2x counterparts
        configs = [
            ("rx_ddim", 1), ("ddim", 2),
            ("rx_ddim", 5), ("ddim", 10),
            ("rx_ddim", 10), ("ddim", 20),
            ("rx_ddim", 25), ("ddim", 50),
            ("rx_ddim", 50), ("ddim", 100),
        ]

        print(f"Launching benchmark: {len(configs)} KID workers + 1 L2 worker")
        print(f"  KID: {kid_num_samples} samples, batch_size={kid_batch_size}, k={kid_k}")
        print(f"  L2:  8 samples, seed={benchmark_seed}")

        # Launch L2 evaluation
        l2_handle = benchmark_l2.spawn(
            checkpoint=checkpoint,
            num_samples_l2=8,
            k=kid_k,
            seed=benchmark_seed,
        )

        # Launch KID workers in parallel
        kid_handles = []
        for method_name, steps in configs:
            h = benchmark_kid_single.spawn(
                method=method_name,
                num_steps=steps,
                checkpoint=checkpoint,
                num_samples=kid_num_samples,
                batch_size=kid_batch_size,
                k=kid_k,
            )
            kid_handles.append((method_name, steps, h))

        # Collect KID results
        print("\nWaiting for KID workers...")
        kid_rows = []
        for method_name, steps, h in kid_handles:
            result_line = h.get()
            kid_rows.append(result_line)
            print(f"  {method_name} steps={steps}: {result_line}")

        # Write kid_vs_nfe.csv to volume
        kid_csv_path = write_kid_csv.remote(kid_rows)
        print(f"KID results written to: {kid_csv_path}")

        # Wait for L2
        l2_path = l2_handle.get()
        print(f"L2 results written to: {l2_path}")

        print("\nBenchmark complete! Download results with:")
        print("  modal volume get cmu-10799-diffusion-data benchmark/kid_vs_nfe.csv .")
        print("  modal volume get cmu-10799-diffusion-data benchmark/l2_vs_nfe.csv .")
    elif action == "evaluate_pd":
        if checkpoint is None:
            raise ValueError("--checkpoint is required for --action evaluate_pd")
        result = evaluate_pd_sweep.remote(
            checkpoint=checkpoint,
            num_samples=num_samples or 5000,
            batch_size=batch_size or 32,
        )
        print(f"PD KID sweep complete. CSV at: {result}")
        print("Download with:")
        print("  modal volume get cmu-10799-diffusion-data benchmark/kid_pd_vs_nfe.csv .")
    elif action == "evaluate_pd_fid":
        if checkpoint is None:
            raise ValueError("--checkpoint is required for --action evaluate_pd_fid")
        result = evaluate_pd_fid.remote(
            checkpoint=checkpoint,
            num_steps=num_steps or 1,
            num_samples=num_samples or 5000,
            batch_size=batch_size or 32,
        )
        print(result)
    elif action == "evaluate_dws":
        if checkpoint is None:
            raise ValueError("--checkpoint (base DDPM) is required for --action evaluate_dws")
        if distilled_checkpoint is None:
            raise ValueError("--distilled-checkpoint (PD model) is required for --action evaluate_dws")
        result = evaluate_dws_sweep.remote(
            base_checkpoint=checkpoint,
            distilled_checkpoint=distilled_checkpoint,
            tau=dws_tau or 0.3,
            k=k or 2,
            extrapolation=extrapolation or "standard",
            num_samples=num_samples or 5000,
            batch_size=batch_size or 32,
        )
        print(f"DWS KID sweep complete. CSV at: {result}")
        print("Download with:")
        print("  modal volume get cmu-10799-diffusion-data benchmark/kid_dws_vs_nfe.csv .")
    elif action == "evaluate_tau":
        if checkpoint is None:
            raise ValueError("--checkpoint (base DDPM) is required for --action evaluate_tau")
        if distilled_checkpoint is None:
            raise ValueError("--distilled-checkpoint (PD model) is required for --action evaluate_tau")
        result = evaluate_tau_sweep.remote(
            base_checkpoint=checkpoint,
            distilled_checkpoint=distilled_checkpoint,
            n_rx_steps=n_rx_steps or 1,
            k=k or 2,
            extrapolation=extrapolation or "standard",
            num_samples=num_samples or 5000,
            batch_size=batch_size or 32,
        )
        print(f"Tau sweep complete. CSV at: {result}")
        print("Download with:")
        print("  modal volume get cmu-10799-diffusion-data benchmark/kid_dws_tau_sweep.csv .")
    elif action == "evaluate_dws_grid":
        if checkpoint is None:
            raise ValueError("--checkpoint (base DDPM) is required for --action evaluate_dws_grid")
        if distilled_checkpoint is None:
            raise ValueError("--distilled-checkpoint (PD model) is required for --action evaluate_dws_grid")
        result = evaluate_dws_grid.remote(
            base_checkpoint=checkpoint,
            distilled_checkpoint=distilled_checkpoint,
            k=k or 2,
            extrapolation=extrapolation or "standard",
            num_samples=num_samples or 5000,
            batch_size=batch_size or 32,
        )
        print(f"DWS grid sweep complete. CSV at: {result}")
        print("Download with:")
        print("  modal volume get cmu-10799-diffusion-data benchmark/kid_dws_grid.csv .")
    else:
        print(f"Unknown action: {action}")
        print("Valid actions: download, train, distill, sample, evaluate, evaluate_pd, evaluate_dws, evaluate_tau, evaluate_dws_grid, benchmark, benchmark_timing")
