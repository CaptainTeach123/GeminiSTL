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
    
    # Filter out corner vignettes and small isolated debris (NO perimeter dilation)
    labeled, num_features = label(raw_mask)
    sizes = np.bincount(labeled.ravel())
    
    clean_mask = np.zeros_like(raw_mask, dtype=bool)
    for lbl in range(1, num_features + 1):
        if sizes[lbl] < 100:
            continue
        comp_lum = lum_f[labeled == lbl].mean()
        comp_g = G_f[labeled == lbl].mean()
        if comp_lum > 30 and (comp_g > 10 or sizes[lbl] > 500):
            clean_mask |= (labeled == lbl)
            
    # Smart internal void healing: seal small reflection crevices (< 1200 px) inside the metal
    # while keeping large genuine tribal negative cutouts (> 1500 px) 100% open
    all_filled = binary_fill_holes(clean_mask)
    holes = all_filled ^ clean_mask
    hole_labeled, hole_num = label(holes)
    hole_sizes = np.bincount(hole_labeled.ravel())
    
    filled_mask = clean_mask.copy()
    for h_lbl in range(1, hole_num + 1):
        if hole_sizes[h_lbl] < 1200:
            filled_mask |= (hole_labeled == h_lbl)
            
    # Gentle 1-pixel closing to seal microscopic edge reflection notches without net boundary expansion
    struct = generate_binary_structure(2, 2)
    final_full_mask = binary_closing(filled_mask, structure=struct, iterations=1)
            
    # 2. Resample clean mask to high-resolution grid
    aspect = final_full_mask.shape[0] / final_full_mask.shape[1]
    target_w = max_size
    target_h = int(target_w * aspect)
    mask_img = Image.fromarray((final_full_mask * 255).astype(np.uint8)).resize(
        (target_w, target_h), Image.Resampling.BILINEAR
    )
    mask_grid = np.array(mask_img) > 128
    
    # Resample luminance for realistic chrome surface sculpting
    img_src = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    arr_src = np.array(img_src, dtype=np.float32)
    lum = (0.299 * arr_src[:,:,0] + 0.587 * arr_src[:,:,1] + 0.114 * arr_src[:,:,2]) / 255.0
    lum_smooth = gaussian_filter(lum, sigma=1.0)
    
    print(f"Generating tapered 3D sculpted geometry on {target_w}x{target_h} grid...")
    
    # 3. Precision Tapered Base & Sculpted Relief:
    # Eliminates rough 90-degree vertical stair-step walls by tapering the base (4px draft angle)
    # and tapers the horns dynamically from edge to crest, matching the example model.
    dist = distance_transform_edt(mask_grid)
    d_max = dist.max() + 1e-5
    norm_dist = dist / d_max
    
    # Smooth tapered base: ramps smoothly from 0 at boundary to base_thickness over 4 pixels
    base_taper = np.clip(dist / 4.0, 0.0, 1.0)
    
    # Continuous tapered sculpted profile: aerodynamic blade taper + surface shading detail
    horn_taper = norm_dist ** 0.85
    height_profile = (horn_taper * 0.75) + (lum_smooth * 0.25 * mask_grid)
    height_profile = gaussian_filter(height_profile, sigma=0.8)
    
    # Total Z height map: tapered base + sculpted relief
    Z = np.where(mask_grid, (base_taper * base_thickness) + (height_profile * bevel_height), 0.0)
    
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
