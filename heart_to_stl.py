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
    base_thickness=3.0,    # Base thickness / center band in mm
    bevel_height=4.5,      # Sculpted bevel / blade ridge height in mm
    double_sided=True,     # Symmetrical double-sided blade: top and bottom are identical and beveled/chamfered
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
    lum_f = (0.299 * R_f + 0.587 * G_f + 0.114 * B_f) / 255.0
    
    # Background rejection (capped at 205 so bright red metallic chrome highlights are preserved)
    is_bg = (R_f > 35) & (R_f <= 205) & (R_f > 1.8 * G_f) & (R_f > 1.8 * B_f) & (G_f < 35) & (B_f < 75)
    raw_mask = ~is_bg
    
    # Filter out corner vignettes and small isolated debris
    labeled, num_features = label(raw_mask)
    sizes = np.bincount(labeled.ravel())
    
    clean_mask = np.zeros_like(raw_mask, dtype=bool)
    for lbl in range(1, num_features + 1):
        if sizes[lbl] < 1000:
            continue
        comp_lum = lum_f[labeled == lbl].mean()
        comp_g = G_f[labeled == lbl].mean()
        if comp_lum > 0.12 and (comp_g > 10 or sizes[lbl] > 5000):
            clean_mask |= (labeled == lbl)
            
    # Smart internal void healing: seal small reflection crevices (< 1200 px) inside the metal
    # while keeping genuine tribal negative cutouts (> 1200 px) 100% open
    all_filled = binary_fill_holes(clean_mask)
    holes = all_filled ^ clean_mask
    hole_labeled, hole_num = label(holes)
    hole_sizes = np.bincount(hole_labeled.ravel())
    
    filled_mask = clean_mask.copy()
    for h_lbl in range(1, hole_num + 1):
        if hole_sizes[h_lbl] < 1200:
            filled_mask |= (hole_labeled == h_lbl)
            
    # 4-pixel closing to seal all edge reflection crevices completely (zero random rifts in topography)
    struct = generate_binary_structure(2, 2)
    final_full_mask = binary_closing(filled_mask, structure=struct, iterations=4)
            
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
    lum_smooth = gaussian_filter(lum, sigma=0.5)
    
    print(f"Generating {'double-sided' if double_sided else 'single-sided'} 3D sculpted geometry on {target_w}x{target_h} grid...")
    
    # 3. Precision Sculpted Double-Sided Blade Profile:
    # 45-degree chamfer bevel (12px) + aerodynamic blade spine + rich metallic chrome relief
    dist = distance_transform_edt(mask_grid)
    d_max = dist.max() + 1e-5
    
    chamfer = np.clip(dist / 12.0, 0.0, 1.0)
    spine = (dist / d_max) ** 0.6
    lum_relief = np.clip((lum_smooth - 0.15) / 0.85, 0.0, 1.0) * mask_grid
    
    sculpted = (chamfer * 0.35) + (spine * 0.25) + (lum_relief * 0.40)
    sculpted = gaussian_filter(sculpted, sigma=0.4)
    
    half_h = (base_thickness + bevel_height) / 2.0 if double_sided else bevel_height
    # Edge bevel feathers down to 0 at perimeter (eliminating rough 90-degree stair-steps)
    edge_feather = np.clip(dist / 2.0, 0.0, 1.0)
    
    if double_sided:
        Z = np.where(mask_grid, edge_feather * sculpted * half_h, 0.0)
    else:
        base_taper = np.clip(dist / 4.0, 0.0, 1.0)
        Z = np.where(mask_grid, (base_taper * base_thickness) + (sculpted * bevel_height), 0.0)
    
    # Pad edges with zeros so geometry closes cleanly
    Z = np.pad(Z, pad_width=1, mode="constant", constant_values=0.0)
    mask_grid = Z > 0
    
    H, W = Z.shape
    scale = width_mm / W
    
    faces = []
    print("Building watertight 3D solid mesh...")
    
    for r in range(H - 1):
        for c in range(W - 1):
            z00, z10 = Z[r, c], Z[r, c+1]
            z01, z11 = Z[r+1, c], Z[r+1, c+1]
            
            # If all 4 corners are 0, skip
            if z00 == 0 and z10 == 0 and z01 == 0 and z11 == 0:
                continue
                
            x0, x1 = c * scale, (c + 1) * scale
            y0, y1 = (H - r) * scale, (H - (r + 1)) * scale
            
            if double_sided:
                # Adaptive diagonal split to avoid diagonal saddle line through zero
                if (z01 == 0 and z10 == 0 and (z00 > 0 or z11 > 0)) or abs(z00 - z11) < abs(z01 - z10):
                    # Tri 1: (00, 01, 11)
                    if not (z00 == 0 and z01 == 0 and z11 == 0):
                        faces.append([[x0, y0, z00], [x0, y1, z01], [x1, y1, z11]])
                        faces.append([[x0, y0, -z00], [x1, y1, -z11], [x0, y1, -z01]])
                    # Tri 2: (00, 11, 10)
                    if not (z00 == 0 and z11 == 0 and z10 == 0):
                        faces.append([[x0, y0, z00], [x1, y1, z11], [x1, y0, z10]])
                        faces.append([[x0, y0, -z00], [x1, y0, -z10], [x1, y1, -z11]])
                else:
                    # Tri 1: (00, 01, 10)
                    if not (z00 == 0 and z01 == 0 and z10 == 0):
                        faces.append([[x0, y0, z00], [x0, y1, z01], [x1, y0, z10]])
                        faces.append([[x0, y0, -z00], [x1, y0, -z10], [x0, y1, -z01]])
                    # Tri 2: (10, 01, 11)
                    if not (z10 == 0 and z01 == 0 and z11 == 0):
                        faces.append([[x1, y0, z10], [x0, y1, z01], [x1, y1, z11]])
                        faces.append([[x1, y0, -z10], [x1, y1, -z11], [x0, y1, -z01]])
            else:
                # Top sculpted surface (+Z)
                faces.append([[x0, y0, z00], [x0, y1, z01], [x1, y0, z10]])
                faces.append([[x1, y0, z10], [x0, y1, z01], [x1, y1, z11]])
                
                # Flat bottom surface (Z=0)
                faces.append([[x0, y0, 0], [x1, y0, 0], [x0, y1, 0]])
                faces.append([[x1, y0, 0], [x1, y1, 0], [x0, y1, 0]])
                
                # Side walls (single-sided mode)
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
