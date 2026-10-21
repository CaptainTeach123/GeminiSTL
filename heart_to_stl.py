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
    max_size=750,          # High resolution for razor-sharp curves and smooth edges
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
    
    aspect = img.height / img.width
    target_w = max_size
    target_h = int(target_w * aspect)
    
    img_res = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    arr_res = np.array(img_res, dtype=np.float32)
    R, G, B = arr_res[:, :, 0], arr_res[:, :, 1], arr_res[:, :, 2]
    lum = (0.299 * R + 0.587 * G + 0.114 * B) / 255.0
    
    # 1. Background rejection (capped at 205 so bright red metallic chrome highlights remain)
    is_bg = (R > 35) & (R <= 205) & (R > 1.8 * G) & (R > 1.8 * B) & (G < 35) & (B < 75)
    raw_mask = ~is_bg
    
    # 2. Filter out corner vignettes and small isolated debris
    lbl, num_lbl = label(raw_mask)
    sizes = np.bincount(lbl.ravel())
    clean_mask = np.zeros_like(raw_mask, dtype=bool)
    for i in range(1, num_lbl + 1):
        if sizes[i] < 500: continue
        comp_lum = lum[lbl == i].mean()
        comp_g = G[lbl == i].mean()
        if comp_lum > 0.15 and (comp_g > 15 or sizes[i] > 10000):
            clean_mask |= (lbl == i)
            
    # Guarantee outer background connectivity around boundaries
    clean_mask[:4, :] = False
    clean_mask[-4:, :] = False
    clean_mask[:, :4] = False
    clean_mask[:, -4:] = False
    
    # 3. Heal small specular reflection voids (< 120 px) while keeping the 4 genuine cutouts open!
    all_filled = binary_fill_holes(clean_mask)
    holes = all_filled ^ clean_mask
    lbl_h, num_h = label(holes)
    h_sizes = np.bincount(lbl_h.ravel())
    filled_mask = clean_mask.copy()
    for i in range(1, num_h + 1):
        if h_sizes[i] < 120:
            filled_mask |= (lbl_h == i)
            
    # 4. Level-set contour smoothing: eliminates all jagged pixel stairsteps
    smooth_float = gaussian_filter(filled_mask.astype(np.float32), sigma=1.8)
    mask_grid = (smooth_float >= 0.5)
    
    print(f"Generating {'double-sided' if double_sided else 'single-sided'} 3D sculpted geometry on {target_w}x{target_h} grid...")
    
    # 5. Distance transform & precision sculpted blade profile
    dist = distance_transform_edt(mask_grid)
    d_max = dist.max() + 1e-5
    
    edge_feather = np.clip(dist / 4.0, 0.0, 1.0) ** 1.2
    alpha = np.clip(smooth_float * 2.0 - 0.5, 0.0, 1.0)
    
    chamfer = np.clip(dist / 14.0, 0.0, 1.0)
    spine = (dist / d_max) ** 0.6
    lum_smooth = gaussian_filter(lum, sigma=0.5)
    lum_relief = np.clip((lum_smooth - 0.15) / 0.85, 0.0, 1.0) * mask_grid
    
    sculpted = (chamfer * 0.35) + (spine * 0.25) + (lum_relief * 0.40)
    sculpted = gaussian_filter(sculpted, sigma=0.4)
    
    half_h = (base_thickness + bevel_height) / 2.0 if double_sided else bevel_height
    if double_sided:
        Z = np.where(mask_grid, edge_feather * sculpted * half_h * alpha, 0.0)
    else:
        base_taper = np.clip(dist / 4.0, 0.0, 1.0)
        Z = np.where(mask_grid, (base_taper * base_thickness) + (sculpted * bevel_height * alpha), 0.0)
    
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
