import shutil
from pathlib import Path

import cupy
import torch
import torchvision as tv
from thop import profile as ops_profile
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader.viton_dataset import LoadVITONDataset
from pipelines import DMVTONPipeline
from opt.test_opt import TestOptions
from utils.general import Profile, print_log, warm_up
from utils.metrics import calculate_fid_given_paths, calculate_lpips_given_paths
from utils.torch_utils import select_device


def run_test_pf(
    pipeline, data_loader, device, img_dir, save_dir, log_path, save_img=True
):
    metrics = {}

    result_dir = Path(save_dir) / "results"
    tryon_dir = result_dir / "tryon"
    visualize_dir = result_dir / "visualize"
    tryon_dir.mkdir(parents=True, exist_ok=True)
    visualize_dir.mkdir(parents=True, exist_ok=True)

    # Warm-up gpu
    dummy_input = {
        "person": torch.randn(1, 3, 256, 192).to(device),
        "clothes": torch.randn(1, 3, 256, 192).to(device),
        "clothes_edge": torch.randn(1, 1, 256, 192).to(device),
    }
    with cupy.cuda.Device(int(device.split(":")[-1])):
        warm_up(pipeline, **dummy_input)

    with torch.no_grad():
        seen, dt = 0, Profile(device=device)

        for idx, data in enumerate(tqdm(data_loader)):
            # Prepare data
            real_image = data["image"].to(device)
            clothes = data["color"].to(device)
            edge = data["edge"].to(device)

            with cupy.cuda.Device(int(device.split(":")[-1])):
                with dt:
                    p_tryon, warped_cloth = pipeline(
                            clothes = transform(garment).unsqueeze(0).to(device)with cupy.cuda.Device(int(device.split(':')[-1])):

import shutil
from pathlib import Path

import cupy
import torch
import torchvision as tv
from thop import profile as ops_profile
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader.viton_dataset import LoadVITONDataset
from pipelines import DMVTONPipeline
from opt.test_opt import TestOptions
from utils.general import Profile, print_log, warm_up
from utils.metrics import calculate_fid_given_paths, calculate_lpips_given_paths
from utils.torch_utils import select_device


def run_test_pf(
    pipeline, data_loader, device, img_dir, save_dir, log_path, save_img=True
):
    metrics = {}

    result_dir = Path(save_dir) / "results"
    tryon_dir = result_dir / "tryon"
    visualize_dir = result_dir / "visualize"
    tryon_dir.mkdir(parents=True, exist_ok=True)
    visualize_dir.mkdir(parents=True, exist_ok=True)

    # Warm-up gpu
    dummy_input = {
        "person": torch.randn(1, 3, 256, 192).to(device),
        "clothes": torch.randn(1, 3, 256, 192).to(device),
        "clothes_edge": torch.randn(1, 1, 256, 192).to(device),
    }
    with cupy.cuda.Device(int(device.split(":")[-1])):
        warm_up(pipeline, **dummy_input)

    with torch.no_grad():
        seen, dt = 0, Profile(device=device)

        for idx, data in enumerate(tqdm(data_loader)):
            # Prepare data
            real_image = data["image"].to(device)
            clothes = data["color"].to(device)
            edge = data["edge"].to(device)

            with cupy.cuda.Device(int(device.split(":")[-1])):
                with dt:
                    p_tryon, warped_cloth = pipeline(
