
# pix2vox.py
# Implementation of an enhanced Pix2Vox model for 3D voxel reconstruction from multi-view 2D images
# Uses PyTorch with ResNet50 backbone, mixed precision training, and advanced data augmentation
# Includes dataset creation, training, evaluation, and visualization functionalities

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from skimage.transform import resize
from torchvision.models import resnet50
from torch.utils.data import Dataset, DataLoader
import gc
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from scipy.ndimage import gaussian_filter, binary_closing, binary_dilation
from torch.cuda.amp import GradScaler
import time
from google.colab import files
from scipy.spatial.distance import cdist

# Set memory allocation configuration for PyTorch to optimize GPU memory usage
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Initialize device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Clean up memory to prevent GPU memory overflow
gc.collect()
torch.cuda.empty_cache()

# MultiViewDataset class for handling multi-view images, point clouds, and voxels
class MultiViewDataset(Dataset):
    def __init__(self, root_dir, transform=None, img_size=224, num_points=2048, voxel_size=64):
        """
        Initialize the dataset with specified parameters.
        
        Args:
            root_dir (str): Directory to store or load dataset
            transform (callable, optional): Transformations to apply to images
            img_size (int): Size to resize images
            num_points (int): Number of points in point cloud
            voxel_size (int): Size of voxel grid (voxel_size x voxel_size x voxel_size)
        """
        self.root_dir = root_dir
        self.num_points = num_points
        self.voxel_size = voxel_size
        # Define image transformations with data augmentation
        self.transform = transform or transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(30),  # Random rotation up to 30 degrees
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.8, 1.2)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            transforms.RandomCrop(img_size, padding=10),  # Random cropping with padding
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.1)),  # Simulate occlusions
        ])
        self.models = self._create_enhanced_sample_data()

    def _create_enhanced_sample_data(self):
        """
        Create synthetic dataset with various 3D shapes and their multi-view images, point clouds, and voxels.
        
        Returns:
            list: List of directories containing sample data
        """
        print("Creating enhanced sample data with realistic shapes...")
        os.makedirs(self.root_dir, exist_ok=True)
        models = []
        shape_types = ['cube', 'sphere', 'cylinder', 'cone', 'torus', 'pyramid', 'prism', 'ellipsoid']

        # Create background directory for synthetic images
        bg_dir = os.path.join(self.root_dir, 'backgrounds')
        os.makedirs(bg_dir, exist_ok=True)
        # Create a dummy background if none exists
        if not os.listdir(bg_dir):
            dummy_bg = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            Image.fromarray(dummy_bg).save(os.path.join(bg_dir, 'dummy_bg.jpg'))

        # Generate 1000 samples with different shapes and views
        for i in range(1000):
            model_dir = os.path.join(self.root_dir, f"sample_{i}")
            os.makedirs(model_dir, exist_ok=True)
            shape_type = shape_types[i % len(shape_types)]

            # Create images for each view
            for view in ['front', 'left', 'right', 'back', 'top']:
                img = self._create_shape_image(256, 256, shape_type, view, bg_dir)
                Image.fromarray(img).save(os.path.join(model_dir, f"{view}.jpg"))

            # Generate and save point cloud
            points = self._create_shape_pointcloud(shape_type)
            np.save(os.path.join(model_dir, "pointcloud.npy"), points)

            # Convert points to voxels and save
            voxels = self._points_to_voxels(points, voxel_size=self.voxel_size)
            np.save(os.path.join(model_dir, "voxels.npy"), voxels)

            models.append(model_dir)
        return models

    def _create_shape_image(self, width, height, shape_type, view, bg_dir):
        """
        Create a synthetic 2D image of a shape on a random background.
        
        Args:
            width (int): Image width
            height (int): Image height
            shape_type (str): Type of shape to generate
            view (str): View angle (front, left, right, back, top)
            bg_dir (str): Directory containing background images
        
        Returns:
            numpy.ndarray: Generated image
        """
        # Load random background
        bg_path = np.random.choice(os.listdir(bg_dir))
        img = np.array(Image.open(os.path.join(bg_dir, bg_path)).resize((width, height)))
        if img.shape[-1] == 4:  # Handle RGBA
            img = img[:, :, :3]

        # Create mask for shape
        mask = np.zeros((height, width), dtype=np.uint8)
        center = (width//2, height//2)
        size = min(width, height) // 3

        # Define color mapping for different shapes
        color_map = {
            'cube': (200, 50, 50),
            'sphere': (50, 200, 50),
            'cylinder': (50, 50, 200),
            'cone': (200, 200, 50),
            'torus': (200, 50, 200),
            'pyramid': (100, 100, 50),
            'prism': (50, 100, 100),
            'ellipsoid': (100, 50, 100)
        }

        # Generate shape-specific mask
        if shape_type == 'cube':
            pts = np.array([
                [center[0]-size, center[1]-size],
                [center[0]+size, center[1]-size],
                [center[0]+size, center[1]+size],
                [center[0]-size, center[1]+size]
            ], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)

        elif shape_type == 'sphere':
            cv2.circle(mask, center, size, 255, -1)

        elif shape_type == 'cylinder':
            cv2.ellipse(mask, center, (size, size//2), 0, 0, 360, 255, -1)

        elif shape_type == 'cone':
            pts = np.array([
                [center[0], center[1]-size],
                [center[0]+size, center[1]+size],
                [center[0]-size, center[1]+size]
            ], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)

        elif shape_type == 'torus':
            cv2.circle(mask, (center[0]-size//2, center[1]), size//3, 255, -1)
            cv2.circle(mask, (center[0]+size//2, center[1]), size//3, 255, -1)

        elif shape_type == 'pyramid':
            pts = np.array([
                [center[0], center[1]-size],
                [center[0]+size, center[1]+size],
                [center[0]-size, center[1]+size]
            ], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)

        elif shape_type == 'prism':
            pts = np.array([
                [center[0]-size, center[1]-size],
                [center[0]+size, center[1]-size],
                [center[0]+int(size*1.5), center[1]],
                [center[0]+size, center[1]+size],
                [center[0]-size, center[1]+size],
                [center[0]-int(size*1.5), center[1]]
            ], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)

        elif shape_type == 'ellipsoid':
            cv2.ellipse(mask, center, (size, size//2), 0, 0, 360, 255, -1)

        # Composite shape onto background
        color = color_map[shape_type]
        img[mask == 255] = color
        cv2.putText(img, f"{shape_type} ({view})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        return img

    def _create_shape_pointcloud(self, shape_type):
        """
        Generate a point cloud for a given shape type.
        
        Args:
            shape_type (str): Type of shape to generate
        
        Returns:
            numpy.ndarray: Point cloud with num_points points
        """
        points = np.zeros((self.num_points, 3))

        if shape_type == 'cube':
            points = np.random.rand(self.num_points, 3) - 0.5
            max_coords = np.argmax(np.abs(points), axis=1)
            for i in range(3):
                mask = max_coords == i
                points[mask, i] = 0.5 * np.sign(points[mask, i])

        elif shape_type == 'sphere':
            phi = np.arccos(1 - 2 * np.random.uniform(0, 1, self.num_points))
            theta = np.random.uniform(0, 2*np.pi, self.num_points)
            r = 0.5
            points[:, 0] = r * np.sin(phi) * np.cos(theta)
            points[:, 1] = r * np.sin(phi) * np.sin(theta)
            points[:, 2] = r * np.cos(phi)

        elif shape_type == 'cylinder':
            height = np.random.uniform(-0.5, 0.5, self.num_points)
            theta = np.random.uniform(0, 2*np.pi, self.num_points)
            r = 0.5
            points[:, 0] = r * np.cos(theta)
            points[:, 1] = height
            points[:, 2] = r * np.sin(theta)

        elif shape_type == 'cone':
            height = np.random.uniform(-0.5, 0.5, self.num_points)
            theta = np.random.uniform(0, 2*np.pi, self.num_points)
            r = 0.5 * (0.5 - height)
            points[:, 0] = r * np.cos(theta)
            points[:, 1] = height
            points[:, 2] = r * np.sin(theta)

        elif shape_type == 'torus':
            theta = np.random.uniform(0, 2*np.pi, self.num_points)
            phi = np.random.uniform(0, 2*np.pi, self.num_points)
            R = 0.5
            r = 0.2
            points[:, 0] = (R + r * np.cos(theta)) * np.cos(phi)
            points[:, 1] = (R + r * np.cos(theta)) * np.sin(phi)
            points[:, 2] = r * np.sin(theta)

        elif shape_type == 'pyramid':
            base_points = np.random.rand(self.num_points//2, 2) - 0.5
            height = np.random.uniform(0, 0.5, self.num_points//2)
            points[:self.num_points//2, 0] = base_points[:, 0] * (0.5 - height)
            points[:self.num_points//2, 1] = base_points[:, 1] * (0.5 - height)
            points[:self.num_points//2, 2] = height
            points[self.num_points//2:] = np.random.rand(self.num_points//2, 3) * [0.5, 0.5, 0.5]

        elif shape_type == 'prism':
            theta = np.random.uniform(0, 2*np.pi, self.num_points)
            r = np.random.uniform(0, 0.5, self.num_points)
            height = np.random.uniform(-0.5, 0.5, self.num_points)
            points[:, 0] = r * np.cos(theta)
            points[:, 1] = r * np.sin(theta)
            points[:, 2] = height

        elif shape_type == 'ellipsoid':
            phi = np.arccos(1 - 2 * np.random.uniform(0, 1, self.num_points))
            theta = np.random.uniform(0, 2*np.pi, self.num_points)
            r_x, r_y, r_z = 0.5, 0.3, 0.4
            points[:, 0] = r_x * np.sin(phi) * np.cos(theta)
            points[:, 1] = r_y * np.sin(phi) * np.sin(theta)
            points[:, 2] = r_z * np.cos(phi)

        return points

    def _points_to_voxels(self, points, voxel_size=64):
        """
        Convert point cloud to voxel grid.
        
        Args:
            points (numpy.ndarray): Input point cloud
            voxel_size (int): Size of voxel grid
        
        Returns:
            numpy.ndarray: Voxel grid
        """
        voxels = np.zeros((voxel_size, voxel_size, voxel_size), dtype=np.float32)
        scaled_points = (points * (voxel_size//2 - 1) + voxel_size//2).astype(int)

        for x, y, z in scaled_points:
            if 0 <= x < voxel_size and 0 <= y < voxel_size and 0 <= z < voxel_size:
                voxels[x, y, z] = 1.0

        # Apply smoothing and morphological operations
        voxels = gaussian_filter(voxels, sigma=0.5)
        voxels = (voxels > 0.1).astype(np.float32)
        voxels = binary_closing(voxels, structure=np.ones((3,3,3))).astype(np.float32)
        return voxels

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.models)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
        
        Returns:
            tuple: (images, targets) where images is a tensor of 5 views and targets contains pointcloud and voxels
        """
        model_path = self.models[idx]
        images = []
        for view in ['front', 'left', 'right', 'back', 'top']:
            img = Image.open(os.path.join(model_path, f"{view}.jpg")).convert('RGB')
            img = self.transform(img)
            images.append(img)
        images = torch.stack(images)

        pc = np.load(os.path.join(model_path, "pointcloud.npy"))
        if len(pc) != self.num_points:
            if len(pc) < self.num_points:
                pad = np.zeros((self.num_points - len(pc), 3))
                pc = np.vstack([pc, pad])
            else:
                indices = np.random.choice(len(pc), self.num_points, replace=False)
                pc = pc[indices]
        pc = torch.from_numpy(pc).float()

        voxels = torch.from_numpy(np.load(os.path.join(model_path, "voxels.npy"))).float()
        return images, {'pointcloud': pc, 'voxels': voxels}

# Enhanced Pix2Vox Model
class EnhancedPix2Vox(nn.Module):
    def __init__(self, voxel_size=64):
        """
        Initialize the Pix2Vox model with ResNet50 backbone and attention mechanism.
        
        Args:
            voxel_size (int): Size of output voxel grid
        """
        super(EnhancedPix2Vox, self).__init__()
        self.voxel_size = voxel_size
        # Use pre-trained ResNet50 as encoder
        base_model = resnet50(weights='IMAGENET1K_V1')
        self.encoder = nn.Sequential(*list(base_model.children())[:-2])
        self.view_pool = nn.AdaptiveAvgPool2d(1)
        # Attention mechanism for view aggregation
        self.view_attention = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 5),
            nn.Softmax(dim=1)
        )
        # Decoder for voxel reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.5),
            nn.Linear(2048, 8*8*8*32),
            nn.ReLU(),
            nn.BatchNorm1d(8*8*8*32),
            nn.Unflatten(1, (32, 8, 8, 8)),
            nn.Conv3d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(16),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv3d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(8),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv3d(8, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(4),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv3d(4, 1, kernel_size=3, padding=1)
            # Outputs logits (no sigmoid)
        )

    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 5, 3, img_size, img_size)
        
        Returns:
            torch.Tensor: Predicted voxel grid
        """
        batch_size = x.size(0)
        view_features = []
        for v in range(5):
            feat = self.encoder(x[:, v, :, :, :])
            pooled = self.view_pool(feat).view(batch_size, -1)
            view_features.append(pooled)
        view_features = torch.stack(view_features, dim=1)
        attention_weights = self.view_attention(view_features.mean(dim=1))
        aggregated = (view_features * attention_weights.unsqueeze(2)).sum(dim=1)
        voxels = self.decoder(aggregated)
        return voxels

# Loss Functions
def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """
    Compute focal loss for imbalanced voxel prediction.
    
    Args:
        pred (torch.Tensor): Predicted logits
        target (torch.Tensor): Ground truth
        alpha (float): Weighting factor for positive class
        gamma (float): Focusing parameter
    
    Returns:
        torch.Tensor: Focal loss value
    """
    pred_sigmoid = torch.sigmoid(pred)
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    pt = torch.where(target == 1, pred_sigmoid, 1 - pred_sigmoid)
    focal_term = (1 - pt) ** gamma
    loss = alpha * focal_term * bce
    return loss.mean()

def iou_loss(pred, target):
    """
    Compute IoU loss for voxel prediction.
    
    Args:
        pred (torch.Tensor): Predicted logits
        target (torch.Tensor): Ground truth
    
    Returns:
        torch.Tensor: IoU loss value
    """
    pred_sigmoid = torch.sigmoid(pred)
    intersection = (pred_sigmoid * target).sum(dim=(1, 2, 3))
    union = (pred_sigmoid + target).sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + 1e-8) / (union + 1e-8)
    return 1 - iou.mean()

# Training Function
def train_pix2vox(model, train_loader, val_loader, epochs=100, lr=0.0005, accum_steps=2):
    """
    Train the Pix2Vox model with gradient accumulation and mixed precision.
    
    Args:
        model (nn.Module): Pix2Vox model
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        epochs (int): Number of training epochs
        lr (float): Learning rate
        accum_steps (int): Number of gradient accumulation steps
    
    Returns:
        tuple: Training losses, validation losses, and metrics history
    """
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()
    writer = SummaryWriter(f'runs/pix2vox_{int(time.time())}')
    best_val_iou = 0.0
    patience = 15
    counter = 0
    train_losses = []
    val_losses = []
    metrics_history = {'accuracy': [], 'iou': []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for i, (images, targets) in enumerate(progress_bar):
            images = images.to(device)
            target = targets['voxels'].to(device)

            with torch.amp.autocast('cuda'):
                outputs = model(images).squeeze(1)
                bce_loss = F.binary_cross_entropy_with_logits(outputs, target)
                outputs_sigmoid = torch.sigmoid(outputs)
                dice_loss = 1 - (2 * (outputs_sigmoid * target).sum() / ((outputs_sigmoid + target).sum() + 1e-8))
                focal = focal_loss(outputs, target)
                iou = iou_loss(outputs, target)
                loss = 0.3 * bce_loss + 0.3 * dice_loss + 0.2 * focal + 0.2 * iou
                loss = loss / accum_steps

            scaler.scale(loss).backward()

            if (i + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                torch.cuda.empty_cache()

            running_loss += loss.item() * accum_steps
            progress_bar.set_postfix({'loss': loss.item() * accum_steps})

        model.eval()
        val_loss = 0.0
        val_metrics = {'accuracy': 0.0, 'iou': 0.0}
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                target = targets['voxels'].to(device)
                with torch.amp.autocast('cuda'):
                    outputs = model(images).squeeze(1)
                    bce_loss = F.binary_cross_entropy_with_logits(outputs, target)
                    outputs_sigmoid = torch.sigmoid(outputs)
                    dice_loss = 1 - (2 * (outputs_sigmoid * target).sum() / ((outputs_sigmoid + target).sum() + 1e-8))
                    focal = focal_loss(outputs, target)
                    iou = iou_loss(outputs, target)
                    loss = 0.3 * bce_loss + 0.3 * dice_loss + 0.2 * focal + 0.2 * iou
                pred_bin = (outputs_sigmoid > 0.5).float()
                val_metrics['accuracy'] += ((pred_bin == target).float().mean().item())
                intersection = (pred_bin * target).sum()
                union = (pred_bin + target).sum() - intersection
                val_metrics['iou'] += (intersection / (union + 1e-8)).item()
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_metrics['accuracy'] /= len(val_loader)
        val_metrics['iou'] /= len(val_loader)

        scheduler.step()
        if val_metrics['iou'] > best_val_iou:
            best_val_iou = val_metrics['iou']
            torch.save(model.state_dict(), 'best_pix2vox.pth')
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        train_losses.append(running_loss / len(train_loader))
        val_losses.append(val_loss)
        metrics_history['accuracy'].append(val_metrics['accuracy'])
        metrics_history['iou'].append(val_metrics['iou'])
        writer.add_scalar('Loss/train', train_losses[-1], epoch)
        writer.add_scalar('Loss/val', val_losses[-1], epoch)
        writer.add_scalar('Metrics/accuracy', val_metrics['accuracy'], epoch)
        writer.add_scalar('Metrics/iou', val_metrics['iou'], epoch)
        print(f'\nEpoch {epoch+1}:')
        print(f'Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')
        print(f'Accuracy: {val_metrics["accuracy"]:.4f}, IoU: {val_metrics["iou"]:.4f}')
        torch.cuda.empty_cache()
    writer.close()
    return train_losses, val_losses, metrics_history

# Post-Processing for Voxels
def post_process_voxels(voxels, sigma=0.5):
    """
    Post-process voxel predictions with smoothing and morphological operations.
    
    Args:
        voxels (numpy.ndarray): Input voxel grid
        sigma (float): Gaussian filter sigma
    
    Returns:
        numpy.ndarray: Processed voxel grid
    """
    if len(voxels.shape) == 4:
        if voxels.shape[0] != 1:
            raise ValueError(f"Expected batch size of 1, got {voxels.shape[0]}")
        voxels = voxels[0]
    smoothed = gaussian_filter(voxels.astype(float), sigma=sigma)
    threshold = np.mean(smoothed) + np.std(smoothed)
    binary = (smoothed > threshold).astype(np.float32)
    binary = binary_closing(binary, structure=np.ones((3,3,3))).astype(np.float32)
    binary = binary_dilation(binary, structure=np.ones((3,3,3))).astype(np.float32)
    return binary

# Chamfer Distance for Evaluation
def chamfer_distance(pred_voxels, gt_voxels):
    """
    Compute Chamfer distance between predicted and ground truth voxel grids.
    
    Args:
        pred_voxels (numpy.ndarray): Predicted voxel grid
        gt_voxels (numpy.ndarray): Ground truth voxel grid
    
    Returns:
        float: Chamfer distance
    """
    pred_points = np.argwhere(pred_voxels > 0.5)
    gt_points = np.argwhere(gt_voxels > 0.5)
    if len(pred_points) == 0 or len(gt_points) == 0:
        return float('inf')
    dist1 = np.min(cdist(pred_points, gt_points), axis=1).mean()
    dist2 = np.min(cdist(gt_points, pred_points), axis=1).mean()
    return dist1 + dist2

# Visualization Function
def visualize_voxels(voxels, save_path=None):
    """
    Visualize voxel grid using Plotly or Matplotlib.
    
    Args:
        voxels (numpy.ndarray): Voxel grid to visualize
        save_path (str, optional): Path to save visualization
    """
    from skimage import measure
    import plotly.graph_objects as go

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    try:
        verts, faces, _, _ = measure.marching_cubes(voxels, level=0.5)
        verts = (verts / voxels.shape[0]) - 0.5

        fig = go.Figure(data=[
            go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color='lightblue',
                opacity=0.8,
                flatshading=True
            )
        ])

        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode='data'
            ),
            width=800,
            height=600,
            title='3D Voxel Reconstruction'
        )

        if save_path:
            html_path = save_path.replace('.png', '.html')
            fig.write_html(html_path)
            print(f"Saved interactive visualization to {html_path}")

        fig.show()

    except Exception as e:
        print(f"Error in visualization: {e}")
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        verts, faces, _, _ = measure.marching_cubes(voxels, level=0.5)
        verts = (verts / voxels.shape[0]) - 0.5
        ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], shade=True)
        ax.set_title('3D Voxel Reconstruction')
        plt.tight_layout()
        plt.show()

# Reconstruction Pipeline
class ReconstructionPipeline:
    def __init__(self, voxel_size=64):
        """
        Initialize the reconstruction pipeline with pre-trained Pix2Vox model.
        
        Args:
            voxel_size (int): Size of output voxel grid
        """
        self.voxel_size = voxel_size
        self.pix2vox = EnhancedPix2Vox(voxel_size=voxel_size).to(device)
        try:
            self.pix2vox.load_state_dict(torch.load('best_pix2vox.pth', map_location=device))
            print("Loaded pre-trained Pix2Vox model")
        except FileNotFoundError:
            print("Warning: Model weights not found. Please train the model first.")
        self.pix2vox.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def select_images(self):
        """
        Prompt user to upload images for each view.
        
        Returns:
            list: Paths to uploaded images
        """
        print("Please upload images for each view (front, left, right, back, top).")
        image_paths = []
        views = ['front', 'left', 'right', 'back', 'top']
        for view in views:
            print(f"Uploading image for {view} view...")
            uploaded = files.upload()
            if not uploaded:
                print(f"No file uploaded for {view} view. Aborting.")
                return None
            file_path = list(uploaded.keys())[0]
            with open(file_path, 'wb') as f:
                f.write(uploaded[file_path])
            image_paths.append(file_path)
        return image_paths

    def process_images(self, image_paths):
        """
        Process input images with transformations.
        
        Args:
            image_paths (list): List of image file paths
        
        Returns:
            tuple: Processed image tensor and original RGB images
        """
        images = []
        rgb_images = []
        views = ['front', 'left', 'right', 'back', 'top']
        for view, img_path in zip(views, image_paths):
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
            img = Image.open(img_path).convert('RGB')
            rgb_images.append(img)
            img = self.transform(img)
            images.append(img)
        return torch.stack(images).unsqueeze(0).to(device), rgb_images

    def reconstruct(self, image_paths=None):
        """
        Perform 3D voxel reconstruction from input images.
        
        Args:
            image_paths (list, optional): List of image file paths
        """
        if image_paths is None:
            image_paths = self.select_images()
            if image_paths is None:
                print("Reconstruction aborted due to missing image selections.")
                return

        images_tensor, rgb_images = self.process_images(image_paths)
        print("\nVisualizing input views:")
        self.visualize_inputs(rgb_images)

        print("\nGenerating 3D voxel reconstruction...")
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                voxels = self.pix2vox(images_tensor).squeeze(1)
                voxels = torch.sigmoid(voxels).cpu().numpy()
            voxels = post_process_voxels(voxels, sigma=0.5)

        print("\n3D Reconstruction Results:")
        visualize_voxels(voxels, save_path="outputs/voxel_reconstruction.png")

        self.save_results(voxels, rgb_images)
        print("\n3D reconstruction complete! Check the output files in the 'outputs' folder.")

    def visualize_inputs(self, rgb_images):
        """
        Visualize input images for all views.
        
        Args:
            rgb_images (list): List of PIL images
        """
        plt.figure(figsize=(15, 3))
        views = ['Front', 'Left', 'Right', 'Back', 'Top']
        for i, (rgb_img, view) in enumerate(zip(rgb_images, views)):
            plt.subplot(1, 5, i+1)
            plt.imshow(rgb_img)
            plt.title(view)
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    def save_results(self, voxels, rgb_images):
        """
        Save reconstruction results to disk.
        
        Args:
            voxels (numpy.ndarray): Reconstructed voxel grid
            rgb_images (list): List of input images
        """
        os.makedirs("outputs", exist_ok=True)
        np.save("outputs/voxels.npy", voxels)
        for i, img in enumerate(rgb_images):
            img.save(f"outputs/input_view_{i}.jpg")
        print("Saved voxel data and input images to 'outputs' folder")

# Model Analysis
def analyze_model_performance(model, loader):
    """
    Analyze model performance with confusion matrix and classification report.
    
    Args:
        model (nn.Module): Trained model
        loader (DataLoader): Data loader for evaluation
    
    Returns:
        float: Voxel accuracy
    """
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            target = targets['voxels'].to(device)
            with torch.amp.autocast('cuda'):
                outputs = model(images).squeeze(1)
                outputs_sigmoid = torch.sigmoid(outputs)
            pred_bin = (outputs_sigmoid > 0.5).float()
            all_preds.append(pred_bin.cpu())
            all_targets.append(target.cpu)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    pred_flat = all_preds.view(-1).numpy()
    target_flat = all_targets.view(-1).numpy()
    pred_bin = (pred_flat > 0.5).astype(int)
    target_bin = (target_flat > 0.5).astype(int)
    cm = confusion_matrix(target_bin, pred_bin)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Empty', 'Occupied'],
                yticklabels=['Empty', 'Occupied'])
    plt.title('Voxel Occupancy Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    print("\nClassification Report:")
    print(classification_report(target_bin, pred_bin,
                              target_names=['Empty', 'Occupied']))
    accuracy = (all_preds == all_targets).float().mean().item()
    print(f"\nOverall Voxel Accuracy: {accuracy:.4f}")
    return accuracy

# Custom Collate Function
def custom_collate(batch):
    """
    Custom collate function for DataLoader to handle dictionary targets.
    
    Args:
        batch (list): List of samples from dataset
    
    Returns:
        tuple: Collated images and targets
    """
    images = torch.stack([item[0] for item in batch])
    targets = {
        'pointcloud': torch.stack([item[1]['pointcloud'] for item in batch]),
        'voxels': torch.stack([item[1]['voxels'] for item in batch])
    }
    return images, targets

# Main Function
def main():
    """Main function to execute the training and reconstruction pipeline."""
    voxel_size = 64
    dataset = MultiViewDataset('/content/shapenet_sample', num_points=2048, voxel_size=voxel_size)
    train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=custom_collate, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=custom_collate, num_workers=2)

    pix2vox = EnhancedPix2Vox(voxel_size=voxel_size)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        pix2vox = nn.DataParallel(pix2vox)
    pix2vox = pix2vox.to(device)

    print("Training Enhanced Pix2Vox...")
    train_losses, val_losses, metrics_history = train_pix2vox(pix2vox, train_loader, val_loader, epochs=100, lr=0.0005, accum_steps=2)

    print("\nTraining Curves:")
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(metrics_history['accuracy'], label='Accuracy')
    plt.plot(metrics_history['iou'], label='IoU')
    plt.title('Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\nPix2Vox Model Analysis:")
    accuracy = analyze_model_performance(pix2vox, val_loader)
    if accuracy < 0.9:
        print(f"Warning: Voxel accuracy ({accuracy:.4f}) is below 90%. Consider training for more epochs or tuning hyperparameters.")

    # Evaluate Chamfer distance
    writer = SummaryWriter(f'runs/pix2vox_eval_{int(time.time())}')
    chamfer_scores = []
    pix2vox.eval()
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            target = targets['voxels'].to(device)
            with torch.amp.autocast('cuda'):
                outputs = pix2vox(images).squeeze(1)
                outputs = torch.sigmoid(outputs).cpu().numpy()
            for pred, gt in zip(outputs, target.cpu().numpy()):
                pred = post_process_voxels(pred)
                score = chamfer_distance(pred, gt)
                chamfer_scores.append(score)
    avg_chamfer = np.mean([s for s in chamfer_scores if s != float('inf')])
    print(f"\nAverage Chamfer Distance: {avg_chamfer:.4f}")
    writer.add_scalar('Metrics/chamfer_distance', avg_chamfer, 0)
    writer.close()

    print("\nPreparing for reconstruction...")
    pipeline = ReconstructionPipeline(voxel_size=voxel_size)
    print("Opening file upload prompts to select images...")
    pipeline.reconstruct()

if __name__ == "__main__":
    main()
