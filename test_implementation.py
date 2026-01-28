"""
Quick test script to verify the DDPM implementation
"""
import torch
from src.models.unet import UNet, create_model_from_config
from src.methods.ddpm import DDPM
import yaml

def test_unet():
    """Test U-Net architecture"""
    print("=" * 60)
    print("Testing U-Net Architecture")
    print("=" * 60)
    
    # Create model
    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=128,
        channel_mult=(1, 2, 2, 4),
        num_res_blocks=2,
        attention_resolutions=[16, 8],
        num_heads=4,
        dropout=0.1,
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,} ({num_params / 1e6:.2f}M)")
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 3, 64, 64)
    t = torch.randint(0, 1000, (batch_size,))
    
    print(f"Input shape: {x.shape}")
    print(f"Timestep shape: {t.shape}")
    
    with torch.no_grad():
        out = model(x, t)
    
    print(f"Output shape: {out.shape}")
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"
    print("✓ U-Net forward pass successful!\n")
    
    return model


def test_ddpm(model):
    """Test DDPM implementation"""
    print("=" * 60)
    print("Testing DDPM Implementation")
    print("=" * 60)
    
    device = torch.device('cpu')
    
    # Create DDPM
    ddpm = DDPM(
        model=model,
        device=device,
        num_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
    )
    
    print(f"Number of timesteps: {ddpm.num_timesteps}")
    print(f"Beta schedule: [{ddpm.beta_start}, {ddpm.beta_end}]")
    
    # Test forward process
    batch_size = 2
    x_0 = torch.randn(batch_size, 3, 64, 64)
    t = torch.randint(0, 1000, (batch_size,))
    
    x_t, noise = ddpm.forward_process(x_0, t)
    print(f"✓ Forward process: {x_0.shape} -> {x_t.shape}")
    
    # Test loss computation
    loss, metrics = ddpm.compute_loss(x_0)
    print(f"✓ Loss computation: {loss.item():.4f}")
    print(f"  Metrics: {metrics}")
    
    # Test reverse process
    with torch.no_grad():
        x_prev = ddpm.reverse_process(x_t, t)
    print(f"✓ Reverse process: {x_t.shape} -> {x_prev.shape}")
    
    # Test sampling (just a few steps to be quick)
    print("\nTesting sampling (2 images, full 1000 steps)...")
    with torch.no_grad():
        samples = ddpm.sample(
            batch_size=2,
            image_shape=(3, 64, 64),
            num_steps=1000,
        )
    print(f"✓ Sampling: {samples.shape}")
    print(f"  Sample range: [{samples.min():.2f}, {samples.max():.2f}]")
    print("✓ DDPM implementation successful!\n")


def test_config():
    """Test loading from config"""
    print("=" * 60)
    print("Testing Config Loading")
    print("=" * 60)
    
    # Load config
    with open('configs/ddpm_babel.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("Config loaded successfully:")
    print(f"  Model base_channels: {config['model']['base_channels']}")
    print(f"  Training batch_size: {config['training']['batch_size']}")
    print(f"  DDPM num_timesteps: {config['ddpm']['num_timesteps']}")
    
    # Create model from config
    model = create_model_from_config(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model created with {num_params:,} parameters")
    
    # Create DDPM from config
    device = torch.device('cpu')
    ddpm = DDPM.from_config(model, config, device)
    print(f"  DDPM created with {ddpm.num_timesteps} timesteps")
    print("✓ Config loading successful!\n")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("DDPM Implementation Test Suite")
    print("=" * 60 + "\n")
    
    try:
        # Test U-Net
        model = test_unet()
        
        # Test DDPM
        test_ddpm(model)
        
        # Test config loading
        test_config()
        
        print("=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        print("\nYour DDPM implementation is ready to train!")
        print("To start training, run:")
        print("  python train.py --method ddpm --config configs/ddpm_babel.yaml")
        print("\nTo test with single batch overfitting first:")
        print("  python train.py --method ddpm --config configs/ddpm_babel.yaml --overfit-single-batch")
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("TEST FAILED!")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
