"""
Methods module for cmu-10799-diffusion.

This module contains implementations of generative modeling methods:
- DDPM (Denoising Diffusion Probabilistic Models)
- FlowMatching (Flow Matching for Generative Modeling)
"""

from .base import BaseMethod
from .ddpm import DDPM
from .flow_matching import FlowMatching

__all__ = [
    'BaseMethod',
    'DDPM',
    'FlowMatching',
]
