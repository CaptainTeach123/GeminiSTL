import os
import numpy as np
from PIL import Image
from scipy.ndimage import (
    distance_transform_edt,
    gaussian_filter,
    label,
    binary_fill_holes
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
    
    # 2. Filter out dark corner vignettes while preserving ALL tribal spikes, hooks, and barbs
    lbl, num_lbl = label(raw_mask)
    sizes = np.bincount(lbl.ravel())
    clean_mask = np.zeros_like(raw_mask, dtype=bool)
    for i in range(1, num_lbl + 1):
        comp_lum = lum[lbl == i].mean()
        comp_g = G[lbl == i].mean()
        comp_b = B[lbl == i].mean()
        # Corner vignette is near pitch-black (lum < 0.08, G < 10). Metal heart features have silver/chrome color
        if comp_lum > 0.12 and (comp_g > 15 or comp_b > 25 or sizes[i] > 1000):
            clean_mask |= (lbl == i)
            
    # Guarantee outer background connectivity around boundaries
    clean_mask[:4, :] = False
    clean_mask[-4:, :] = False
    clean_mask[:, :4] = False
    clean_mask[:, -4:] = False
    
    # 3. Heal small specular reflection voids (< 120 px) while keeping all 4 genuine cutouts 100% open!
    all_filled = binary_fill_holes(clean_mask)
    holes = all_filled ^ clean_mask
    lbl_h, num_h = label(holes)
    h_sizes = np.bincount(lbl_h.ravel())
    filled_mask = clean_mask.copy()
    for i in range(1, num_h + 1):
        if h_sizes[i] < 120:
            filled_mask |= (lbl_h == i)
            
    # 4. Continuous Level-Set contour field
    smooth_float = gaussian_filter(filled_mask.astype(np.float32), sigma=1.8)
    mask_grid = (smooth_float >= 0.5)
    
    # 5. Sub-pixel Level-Set gradient projection: eliminates the 90-degree integer raster staircasing
    gy, gx = np.gradient(smooth_float)
    grad_sq = gx**2 + gy**2 + 1e-6
    shift_x = np.clip((0.5 - smooth_float) * gx / grad_sq, -0.85, 0.85)
    shift_y = np.clip((0.5 - smooth_float) * gy / grad_sq, -0.85, 0.85)
    near_bnd = (smooth_float >= 0.08) & (smooth_float <= 0.92)
    shift_x = np.where(near_bnd, shift_x, 0.0)
    shift_y = np.where(near_bnd, shift_y, 0.0)
    
    print(f"Generating {'double-sided' if double_sided else 'single-sided'} 3D sculpted geometry on {target_w}x{target_h} grid...")
    
    # 6. Distance transform & Multi-scale Chiseled Blade Sculpting
    dist = distance_transform_edt(mask_grid)
    d_max = dist.max() + 1e-5
    
    # Knife-edge feather ramp: drops smoothly to zero at perimeter
    edge_feather = np.clip(dist / 3.0, 0.0, 1.0) ** 1.1
    alpha = np.clip(smooth_float * 2.0 - 0.5, 0.0, 1.0)
    
    # Local 45-degree chamfer bevel (steep knife edge)
    chamfer = np.clip(dist / 8.0, 0.0, 1.0)
    
    # Aerodynamic continuous blade roof (ensures narrow spikes reach proud height)
    blade_roof = 1.0 - np.exp(-dist / 8.0)
    
    # Unsharp-masked chrome relief: preserves high-frequency grooves & reflection crests
    lum_base = gaussian_filter(lum, sigma=0.8)
    lum_detail = lum - gaussian_filter(lum, sigma=3.5)
    chiseled_lum = np.clip(lum_base + 1.2 * lum_detail, 0.0, 1.0)
    lum_relief = np.clip((chiseled_lum - 0.12) / 0.88, 0.0, 1.0) * mask_grid
    
    # Composite sculpted topography
    sculpted = (chamfer * 0.30) + (blade_roof * 0.30) + (lum_relief * 0.40)
    sculpted = gaussian_filter(sculpted, sigma=0.3)
    
    half_h = (base_thickness + bevel_height) / 2.0 if double_sided else bevel_height
    if double_sided:
        Z = np.where(mask_grid, edge_feather * sculpted * half_h * alpha, 0.0)
    else:
        base_taper = np.clip(dist / 4.0, 0.0, 1.0)
        Z = np.where(mask_grid, (base_taper * base_thickness) + (sculpted * bevel_height * alpha), 0.0)
        
    # Pad edges with zeros so geometry closes cleanly
    Z = np.pad(Z, pad_width=1, mode="constant", constant_values=0.0)
    shift_x = np.pad(shift_x, pad_width=1, mode="constant", constant_values=0.0)
    shift_y = np.pad(shift_y, pad_width=1, mode="constant", constant_values=0.0)
    mask_grid = Z > 0
    
    H, W = Z.shape
    scale = width_mm / W
    
    faces = []
    print("Building watertight 3D solid mesh with smooth subpixel boundary alignment...")
    
    for r in range(H - 1):
        for c in range(W - 1):
            z00, z10 = Z[r, c], Z[r, c+1]
            z01, z11 = Z[r+1, c], Z[r+1, c+1]
            
            # If all 4 corners are 0, skip
            if z00 == 0 and z10 == 0 and z01 == 0 and z11 == 0:
                continue
                
            # Subpixel level-set vertex positions
            x00 = (c + shift_x[r, c]) * scale
            y00 = (H - (r + shift_y[r, c])) * scale
            
            x10 = (c + 1 + shift_x[r, c+1]) * scale
            y10 = (H - (r + shift_y[r, c+1])) * scale
            
            x01 = (c + shift_x[r+1, c]) * scale
            y01 = (H - (r + 1 + shift_y[r+1, c])) * scale
            
            x11 = (c + 1 + shift_x[r+1, c+1]) * scale
            y11 = (H - (r + 1 + shift_y[r+1, c+1])) * scale
            
            if double_sided:
                # Adaptive diagonal split to avoid diagonal saddle line through zero
                if (z01 == 0 and z10 == 0 and (z00 > 0 or z11 > 0)) or abs(z00 - z11) < abs(z01 - z10):
                    # Tri 1: (00, 01, 11)
                    if not (z00 == 0 and z01 == 0 and z11 == 0):
                        faces.append([[x00, y00, z00], [x01, y01, z01], [x11, y11, z11]])
                        faces.append([[x00, y00, -z00], [x11, y11, -z11], [x01, y01, -z01]])
                    # Tri 2: (00, 11, 10)
                    if not (z00 == 0 and z11 == 0 and z10 == 0):
                        faces.append([[x00, y00, z00], [x11, y11, z11], [x10, y10, z10]])
                        faces.append([[x00, y00, -z00], [x10, y10, -z10], [x11, y11, -z11]])
                else:
                    # Tri 1: (00, 01, 10)
                    if not (z00 == 0 and z01 == 0 and z10 == 0):
                        faces.append([[x00, y00, z00], [x01, y01, z01], [x10, y10, z10]])
                        faces.append([[x00, y00, -z00], [x10, y10, -z10], [x01, y01, -z01]])
                    # Tri 2: (10, 01, 11)
                    if not (z10 == 0 and z01 == 0 and z11 == 0):
                        faces.append([[x10, y10, z10], [x01, y01, z01], [x11, y11, z11]])
                        faces.append([[x10, y10, -z10], [x11, y11, -z11], [x01, y01, -z01]])
            else:
                # Top sculpted surface (+Z)
                faces.append([[x00, y00, z00], [x01, y01, z01], [x10, y10, z10]])
                faces.append([[x10, y10, z10], [x01, y01, z01], [x11, y11, z11]])
                
                # Flat bottom surface (Z=0)
                faces.append([[x00, y00, 0], [x10, y10, 0], [x01, y01, 0]])
                faces.append([[x10, y10, 0], [x11, y11, 0], [x01, y01, 0]])
                
                # Side walls (single-sided mode)
                if r == 0 or not mask_grid[r-1, c]:
                    faces.append([[x00, y00, 0], [x00, y00, z00], [x10, y10, z10]])
                    faces.append([[x00, y00, 0], [x10, y10, z10], [x10, y10, 0]])
                if r == H - 2 or not mask_grid[r+1, c]:
                    faces.append([[x01, y01, 0], [x11, y11, z11], [x01, y01, z01]])
                    faces.append([[x01, y01, 0], [x11, y11, 0], [x11, y11, z11]])
                if c == 0 or not mask_grid[r, c-1]:
                    faces.append([[x00, y00, 0], [x01, y01, z01], [x00, y00, z00]])
                    faces.append([[x00, y00, 0], [x01, y01, 0], [x01, y01, z01]])
                if c == W - 2 or not mask_grid[r, c+1]:
                    faces.append([[x10, y10, 0], [x10, y10, z10], [x11, y11, z11]])
                    faces.append([[x10, y10, 0], [x11, y11, z11], [x11, y11, 0]])

    faces_arr = np.array(faces, dtype=np.float32)
    heart_mesh = mesh.Mesh(np.zeros(faces_arr.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces_arr):
        heart_mesh.vectors[i] = f
        
    heart_mesh.save(output_stl)
    print(f"Success! Saved {output_stl} ({len(faces_arr):,} watertight triangles).")

if __name__ == "__main__":
    create_heart_stl()
