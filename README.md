# CROP-TYPE SEGMENTATION USING SENTINEL-2 BANDS
## Pixel-Level Classification of Wheat and Mustard in U.P

> **Running / deploying the web app:** see [DEPLOY.md](DEPLOY.md) (Docker Compose, locally or on EC2).

A two-stage deep learning pipeline to classify wheat and mustard at 10m resolution using Sentinel-2 satellite imagery, tackling extreme class imbalance where crop pixels are <0.1% of the data.


## Overview

This project implements a two-stage segmentation architecture to identify wheat and mustard crops at pixel level from early-season Sentinel-2 data. The core challenge: 99.9% of pixels are background, causing standard models to fail by predicting "background" everywhere.

**Study Area**: U.P, India (Tile T44RPP)  
**Resolution**: 10m (10980 × 10980 pixels per tile)  
**Crops**: Wheat , Mustard 




## Dataset

### Satellite Data
| Source | Sentinel-2 (ESA Copernicus) |
|--------|-----------------------------|
| Bands | B02 (blue), B03 (green), B04 (red), B08 (NIR) |
| Tile size | 256×256 patches from 10980×10980 scene |
| Training tiles | 8 tiles (256×256 each) |

### Class Distribution
| Tile | Wheat (pixels) | Mustard (pixels) | Background | Type |
|------|----------------|-------------------|------------|------|
| 09_37 | 0 | 0 | 65,536 | Empty |
| 09_38 | 0 | 0 | 65,536 | Empty |
| 10_38 | 112 | 46 | 65,378 | Sparse |
| 10_37 | 0 | 0 | 65,536 | Empty |
| **11_37** | **953** | **218** | **64,365** | **Primary crop tile** |
| 11_38 | 0 | 0 | 65,536 | Empty |
| 12_37 | 0 | 0 | 65,536 | Empty |
| 12_38 | 0 | 0 | 65,536 | Empty |

**The Imbalance**: Crop pixels = 0.1% of total data | Background = 99.9%




## Architecture (Two-Stage Pipeline)

### Stage 1: Crop Detector CNN
Input (256×256×4)
↓

Conv2D(32,3)  →  BatchNorm  →  MaxPool(2)

↓

Conv2D(64,3)  →  BatchNorm  →  MaxPool(2)

↓

Conv2D(128,3)  →  BatchNorm

↓

GlobalAvgPool  →  Dense(64)  →  Dropout(0.3)

↓

Dense(1, sigmoid)

↓

[0 = empty | 1 = crop present]



Loss Function:- BCE (Binary Cross Entropy) Loss : -[y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)]


### Stage 2: U-Net Segmentor

Encoder :- (256X256X4 -> 32X32X256) (4 BANDS x (65536 PIXELS PER TILE)



Decoder:- (32X32X256 -> 256X256X3) (3 CLASSES x (65536 PIXELS PER TILE)



Loss Function:- SCCE (Sparse Categorical Cross-Entropy) Loss:

  For a single pixel:
  
  Loss = -log(y_pred[true_class])
  
  Where:
  - true_class = 0 (background), 1 (wheat), or 2 (mustard)
    
  - y_pred = [p(bg), p(wheat), p(mustard)]




## Key Techniques

### 1. Patch-Based Training
  #  Extract patches centered on crop pixels only
  ```python
  def extract_crop_patches(image, mask, patch_size=64, n_patches=200):
     
      crop_y, crop_x = np.where(mask > 0)
      patches = []
      for _ in range(n_patches):
          idx = np.random.randint(len(crop_y))
          cy, cx = crop_y[idx], crop_x[idx]
          y1 = max(0, cy - patch_size//2)
          x1 = max(0, cx - patch_size//2)
          patches.append(image[y1:y1+patch_size, x1:x1+patch_size])
      return np.array(patches)
```


### 2. Class Weighting:-
  # Force model to care about rare classes
  class_weights = {
      0: 1.0,    # Background
      1: 10.0,   # Wheat (10× importance)
      2: 20.0    # Mustard (20× importance)
  }



  
## Results


### SUMMARY STATISTICS

Average Wheat IoU: 0.0751 ± 0.1911

Average Mustard IoU: 0.0666 ± 0.1761

Average Background IoU: 0.9968 ± 0.0019

Average Mean IoU: 0.3795 ± 0.1218


### CROP TILE ONLY (rasterlayer_11_37.tif)

Wheat - IoU: 0.5804, Dice: 0.7345, F1: 0.7345

Mustard - IoU: 0.5326, Dice: 0.6950, F1: 0.6950

Mean IoU: 0.7018




## Key Findings
  ### 1. The Imbalance Problem is Solved
  
  Standard models predict all background (99% accuracy, 0% useful). Two-stage approach achieves 0.70 mean IoU on actual crop tiles.

  
  ### 2. Spatial Accuracy is High
  
  Pattern correlation: 0.70
  Correct wheat pixels: 711/953 (75%)
  Correct mustard pixels: 147/218 (67%)

  
  ### 3. False Positives are Minimal
  
  Empty tiles: 0 false positives
  Crop tile: Only 491 false positives (0.7% of background)



  
## Interpretation
The model successfully identifies where crops are and which crop type with high confidence. The remaining errors stem from:

1. Mixed pixels at field boundaries

2. Spectral similarity between young wheat and bare soil

3. Limited training examples for mustard (218 pixels only)

With more diverse training tiles, performance would improve further.




## Tech Stack

Component	Tools
  Data:- Sentinel-2 (ESA Copernicus), QGIS, GDAL, rasterio
  
  Processing:- Python, numpy, pandas
  
  Deep Learning:- TensorFlow, Keras
  
  Architecture:- U-Net, CNN
  
  Visualization:- matplotlib, seaborn
  
  Metrics:- IoU (Intersection Over Union) , Dice, Precision, Recall, F1

