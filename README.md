# GeminiSTL

Convert 2D images to 3D watertight STL models for 3D printing.

## Features
- Mask-based extraction isolating chrome / metallic features from dark backgrounds.
- Euclidean distance transform (`distance_transform_edt`) creates a smooth, contoured bevel profile.
- Luminance shading blending for fine embossed surface details.
- Generates a watertight solid mesh (top sculpted surface, flat base, and closed side walls).

## Setup

1. Install Python 3.10+ (if not already installed).
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Place your input image (`high_res.png`) in the root directory and run:

```bash
python heart_to_stl.py
```

The script will generate `tribal_heart.stl` ready for slicing and 3D printing.

