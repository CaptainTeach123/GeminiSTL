import os
import numpy as np
from PIL import Image
from scipy.ndimage import (
    distance_transform_edt,
    gaussian_filter,
    label,
    binary_closing,
    binary_fill_holes,
    generate_binary_structure
)
from stl import mesh

def create_heart_stl(
    image_path="high_res.png",
    output_stl="tribal_heart.stl",
    max_size=650,          # High resolution for razor-sharp curves and smooth edges
    width_mm=120.0,        # Physical width of the 3D print in millimeters
    base_thickness=3.0,    # Solid base thickness in mm
    bevel_height=4.5,      # Chiseled blade ridge height in mm
):
    if not os.path.isabs(image_path):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(base_dir, image_path)
    if not os.path.isabs(output_stl):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_stl = os.path.join(base_dir, output_stl)

    print(f"Loading {image_path}...")
    img = Image.open(image_path).convert("RGB")
    
    # 1. Full-resolution chromatic segmentation:
    # Reject red grunge canvas: R > 35, R <= 205, G <= 35, B <= 75, R > 1.8*G & R > 1.8*B
    arr_full = np.array(img, dtype=np.float32)
    R_f, G_f, B_f = arr_full[:, :, 0], arr_full[:, :, 1], arr_full[:, :, 2]
    lum_f = 0.299 * R_f + 0.587 * G_f + 0.114 * B_f
    
    # Background rejection (capped at 205 so bright red metallic chrome highlights are preserved)
    is_bg = (R_f > 35) & (R_f <= 205) & (R_f > 1.8 * G_f) & (R_f > 1.8 * B_f) & (G_f < 35) & (B_f < 75)
    raw_mask = ~is_bg
    
    # Filter out corner vignettes and small isolated debris
    labeled, num_features = label(raw_mask)
    sizes = np.bincount(labeled.ravel())
    
    clean_mask = np.zeros_like(raw_mask, dtype=bool)
    for lbl in range(1, num_features + 1):
        if sizes[lbl] < 500:
            continue
        comp_lum = lum_f[labeled == lbl].mean()
        comp_g = G_f[labeled == lbl].mean()
        if comp_lum > 30 and comp_g > 15:
            clean_mask |= (labeled == lbl)
            
    # Bridge and heal metallic reflection crevices so the horns remain uninterrupted solid blades
    struct = generate_binary_structure(2, 2)
    mask_closed = binary_closing(clean_mask, structure=struct, iterations=5)
    
    # Smart hole fill: heal internal black reflection gaps, while keeping red background cutouts open
    all_filled = binary_fill_holes(mask_closed)
    holes = all_filled ^ mask_closed
    hole_labeled, hole_num = label(holes)
    final_full_mask = mask_closed.copy()
    for h_lbl in range(1, hole_num + 1):
        h_pix = arr_full[hole_labeled == h_lbl]
        mean_r = h_pix[:, 0].mean()
        mean_g = h_pix[:, 1].mean()
        if mean_r < 45 and mean_g < 25:
            final_full_mask |= (hole_labeled == h_lbl)
            
    # 2. Resample clean mask to high-resolution grid
    aspect = final_full_mask.shape[0] / final_full_mask.shape[1]
    target_w = max_size
    target_h = int(target_w * aspect)
    mask_img = Image.fromarray(final_full_mask.astype(np.uint8) * 255).resize(
        (target_w, target_h), Image.Resampling.BILINEAR
    )
    mask_grid = np.array(mask_img) > 128
    
    print(f"Generating chiseled 3D blade geometry on {target_w}x{target_h} grid...")
    
    # 3. Chiseled tribal blade ridge profile:
    # A power-law curve (p=0.55) produces sharp, prominent 3D blade crests that flow
    # down the center of every horn, swirl, and barb, eliminating flat plateaus and dents.
    dist = distance_transform_edt(mask_grid)
    max_d = dist.max() + 1e-5
    chiseled_profile = (dist / max_d) ** 0.55
    chiseled_profile = gaussian_filter(chiseled_profile, sigma=1.0)
    
    # Total Z height map
    Z = np.where(mask_grid, base_thickness + (chiseled_profile * bevel_height), 0.0)
    
    # Pad edges with zeros so side walls seal cleanly
    Z = np.pad(Z, pad_width=1, mode="constant", constant_values=0.0)
    mask_grid = Z > 0
    
    H, W = Z.shape
    scale = width_mm / W
    
    faces = []
    print("Building watertight 3D mesh...")
    
    for r in range(H - 1):
        for c in range(W - 1):
            if mask_grid[r, c] or mask_grid[r+1, c] or mask_grid[r, c+1] or mask_grid[r+1, c+1]:
                x0, x1 = c * scale, (c + 1) * scale
                y0, y1 = (H - r) * scale, (H - (r + 1)) * scale
                
                z00, z10 = Z[r, c], Z[r, c+1]
                z01, z11 = Z[r+1, c], Z[r+1, c+1]
                
                # Top sculpted surface
                faces.append([[x0, y0, z00], [x0, y1, z01], [x1, y0, z10]])
                faces.append([[x1, y0, z10], [x0, y1, z01], [x1, y1, z11]])
                
                # Flat bottom surface (Z=0)
                faces.append([[x0, y0, 0], [x1, y0, 0], [x0, y1, 0]])
                faces.append([[x1, y0, 0], [x1, y1, 0], [x0, y1, 0]])
                
                # Watertight side walls
                if r == 0 or not mask_grid[r-1, c]:
                    faces.append([[x0, y0, 0], [x0, y0, z00], [x1, y0, z10]])
                    faces.append([[x0, y0, 0], [x1, y0, z10], [x1, y0, 0]])
                if r == H - 2 or not mask_grid[r+1, c]:
                    faces.append([[x0, y1, 0], [x1, y1, z11], [x0, y1, z01]])
                    faces.append([[x0, y1, 0], [x1, y1, 0], [x1, y1, z11]])
                if c == 0 or not mask_grid[r, c-1]:
                    faces.append([[x0, y0, 0], [x0, y1, z01], [x0, y0, z00]])
                    faces.append([[x0, y0, 0], [x0, y1, 0], [x0, y1, z01]])
                if c == W - 2 or not mask_grid[r, c+1]:
                    faces.append([[x1, y0, 0], [x1, y0, z10], [x1, y1, z11]])
                    faces.append([[x1, y0, 0], [x1, y1, z11], [x1, y1, 0]])

    faces_arr = np.array(faces, dtype=np.float32)
    heart_mesh = mesh.Mesh(np.zeros(faces_arr.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces_arr):
        heart_mesh.vectors[i] = f
        
    heart_mesh.save(output_stl)
    print(f"Success! Saved {output_stl} ({len(faces_arr):,} watertight triangles).")

if __name__ == "__main__":
    create_heart_stl()
