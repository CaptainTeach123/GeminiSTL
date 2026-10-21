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
    max_size=800,          # High resolution for silky smooth CAD curves
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
    smooth_float = gaussian_filter(filled_mask.astype(np.float32), sigma=2.0)
    T = 0.5
    mask_grid = (smooth_float >= T)
    
    print(f"Generating {'double-sided' if double_sided else 'single-sided'} Marching Squares CAD geometry on {target_w}x{target_h} grid...")
    
    # 5. Continuous distance transform (sigma=1.5 removes Voronoi grid discretization)
    raw_dist = distance_transform_edt(mask_grid)
    dist = gaussian_filter(raw_dist, sigma=1.5)
    
    # 6. Pure Geometric CAD Blade Sculpting:
    # Direct sinusoidal chamfer bevel rising continuously from perimeter edge (d=0, Z=0)
    # Zero flat brim/shelf, zero artificial feathering
    t_bevel = np.clip(dist / 14.0, 0.0, 1.0)
    bevel_profile = np.sin(t_bevel * (np.pi / 2.0))
    blade_roof = 1.0 - np.exp(-dist / 8.0)
    sculpted = (bevel_profile * 0.70) + (blade_roof * 0.30)
    
    half_h = (base_thickness + bevel_height) / 2.0 if double_sided else bevel_height
    
    if double_sided:
        Z = np.where(mask_grid, sculpted * half_h, 0.0)
    else:
        base_taper = np.clip(dist / 4.0, 0.0, 1.0)
        Z = np.where(mask_grid, (base_taper * base_thickness) + (sculpted * bevel_height), 0.0)
        
    H, W = smooth_float.shape
    scale = width_mm / W
    
    # 7. Sub-pixel Marching Squares edge crossings
    h_cross = (smooth_float[:, :-1] >= T) != (smooth_float[:, 1:] >= T)
    h_x = np.zeros((H, W - 1), dtype=np.float32)
    r_idx, c_idx = np.where(h_cross)
    v0 = smooth_float[r_idx, c_idx]
    v1 = smooth_float[r_idx, c_idx + 1]
    t = np.clip((T - v0) / (v1 - v0 + 1e-12), 0.0001, 0.9999)
    h_x[r_idx, c_idx] = (c_idx + t) * scale

    v_cross = (smooth_float[:-1, :] >= T) != (smooth_float[1:, :] >= T)
    v_y = np.zeros((H - 1, W), dtype=np.float32)
    r_idx, c_idx = np.where(v_cross)
    v0 = smooth_float[r_idx, c_idx]
    v1 = smooth_float[r_idx + 1, c_idx]
    t = np.clip((T - v0) / (v1 - v0 + 1e-12), 0.0001, 0.9999)
    v_y[r_idx, c_idx] = (H - 1 - (r_idx + t)) * scale

    b0 = mask_grid[:-1, :-1].astype(np.uint8)
    b1 = mask_grid[:-1, 1:].astype(np.uint8)
    b2 = mask_grid[1:, 1:].astype(np.uint8)
    b3 = mask_grid[1:, :-1].astype(np.uint8)
    case_grid = b0 | (b1 << 1) | (b2 << 2) | (b3 << 3)

    top_triangles = []
    nz_r, nz_c = np.where(case_grid > 0)
    print(f"Triangulating {len(nz_r):,} active cells with sub-pixel level-set contours...")

    for r, c in zip(nz_r, nz_c):
        code = case_grid[r, c]
        x0 = c * scale
        x1 = (c + 1) * scale
        y_bot = (H - 1 - (r + 1)) * scale
        y_top = (H - 1 - r) * scale
        
        TL = (x0, y_top, Z[r, c])
        TR = (x1, y_top, Z[r, c + 1])
        BR = (x1, y_bot, Z[r + 1, c + 1])
        BL = (x0, y_bot, Z[r + 1, c])
        
        if code == 15:
            top_triangles.append((BL, BR, TR))
            top_triangles.append((BL, TR, TL))
            continue
            
        Et = (h_x[r, c], y_top, 0.0) if (code & 3) in (1, 2) else None
        Er = (x1, v_y[r, c + 1], 0.0) if (code & 6) in (2, 4) else None
        Eb = (h_x[r + 1, c], y_bot, 0.0) if (code & 12) in (4, 8) else None
        El = (x0, v_y[r, c], 0.0) if (code & 9) in (1, 8) else None
        
        poly = None
        if code == 1: poly = [TL, El, Et]
        elif code == 2: poly = [TR, Et, Er]
        elif code == 4: poly = [BR, Er, Eb]
        elif code == 8: poly = [BL, Eb, El]
        elif code == 3: poly = [TL, El, Er, TR]
        elif code == 6: poly = [TR, Et, Eb, BR]
        elif code == 12: poly = [BR, Er, El, BL]
        elif code == 9: poly = [BL, Eb, Et, TL]
        elif code == 14: poly = [BL, BR, TR, Et, El]
        elif code == 13: poly = [BL, BR, Er, Et, TL]
        elif code == 11: poly = [BL, Eb, Er, TR, TL]
        elif code == 7: poly = [El, Eb, BR, TR, TL]
        elif code == 5:
            top_triangles.append((TL, El, Et))
            top_triangles.append((BR, Er, Eb))
            continue
        elif code == 10:
            top_triangles.append((TR, Et, Er))
            top_triangles.append((BL, Eb, El))
            continue

        if poly is not None:
            for i in range(1, len(poly) - 1):
                top_triangles.append((poly[0], poly[i], poly[i + 1]))

    # 8. Symmetrical double-sided solid closure (0 open boundary edges)
    faces = []
    if double_sided:
        for p0, p1, p2 in top_triangles:
            faces.append([p0, p1, p2])
            faces.append([[p0[0], p0[1], -p0[2]], [p2[0], p2[1], -p2[2]], [p1[0], p1[1], -p1[2]]])
    else:
        for p0, p1, p2 in top_triangles:
            faces.append([p0, p1, p2])
            faces.append([[p0[0], p0[1], 0.0], [p2[0], p2[1], 0.0], [p1[0], p1[1], 0.0]])

    faces_arr = np.array(faces, dtype=np.float32)
    heart_mesh = mesh.Mesh(np.zeros(faces_arr.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces_arr):
        heart_mesh.vectors[i] = f
        
    heart_mesh.save(output_stl)
    print(f"Success! Saved {output_stl} ({len(faces_arr):,} watertight triangles).")

if __name__ == "__main__":
    create_heart_stl()
