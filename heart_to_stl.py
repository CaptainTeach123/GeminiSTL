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
    max_size=450,          # High resolution for smooth curves without stepping
    width_mm=120.0,        # Physical width of the 3D print in millimeters
    base_thickness=3.0,    # Solid base thickness in mm
    bevel_height=4.5,      # Additional sculpted bevel height in mm
    bevel_radius_px=9.0,   # Transition width for smooth organic pillowed bevel
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
    # Background is textured red grunge (high R, low G <= 35, low B <= 75, R > 1.8*G & R > 1.8*B).
    # Heart contains chrome (balanced R≈G≈B), neon pink (high B >= 70, R >= 110), and dark reflections.
    arr_full = np.array(img, dtype=np.float32)
    R_f, G_f, B_f = arr_full[:, :, 0], arr_full[:, :, 1], arr_full[:, :, 2]
    lum_f = 0.299 * R_f + 0.587 * G_f + 0.114 * B_f
    
    is_bg = (R_f > 35) & (R_f > 1.8 * G_f) & (R_f > 1.8 * B_f) & (G_f < 35) & (B_f < 75)
    raw_mask = ~is_bg
    
    # Discard corner vignette and isolated background noise using connected components
    labeled, num_features = label(raw_mask)
    sizes = np.bincount(labeled.ravel())
    
    clean_mask = np.zeros_like(raw_mask, dtype=bool)
    for lbl in range(1, num_features + 1):
        if sizes[lbl] < 1000:
            continue
        comp_lum = lum_f[labeled == lbl].mean()
        comp_g = G_f[labeled == lbl].mean()
        # Keep bright chrome/metallic components, drop dark corner vignette (lum < 30)
        if comp_lum > 35 and comp_g > 15:
            clean_mask |= (labeled == lbl)
            
    # Bridge internal metallic reflection crevices inside the solid horns
    struct = generate_binary_structure(2, 2)
    mask_closed = binary_closing(clean_mask, structure=struct, iterations=3)
    
    # Smart hole fill: close metallic reflection gaps, but keep genuine tribal negative cutouts
    all_filled = binary_fill_holes(mask_closed)
    holes = all_filled ^ mask_closed
    hole_labeled, hole_num = label(holes)
    final_full_mask = mask_closed.copy()
    for h_lbl in range(1, hole_num + 1):
        h_pix = arr_full[hole_labeled == h_lbl]
        mean_r = h_pix[:, 0].mean()
        mean_g = h_pix[:, 1].mean()
        # If the hole is a dark reflection gap, fill it. If it has red background, keep it open!
        if mean_r < 45 and mean_g < 25:
            final_full_mask |= (hole_labeled == h_lbl)
            
    # 2. Resample clean mask to target grid resolution
    aspect = final_full_mask.shape[0] / final_full_mask.shape[1]
    target_w = max_size
    target_h = int(target_w * aspect)
    mask_img = Image.fromarray(final_full_mask.astype(np.uint8) * 255).resize(
        (target_w, target_h), Image.Resampling.BILINEAR
    )
    mask_grid = np.array(mask_img) > 128
    
    print(f"Generating smooth sculpted relief on {target_w}x{target_h} grid...")
    
    # 3. Organic pillowed bevel:
    # Uses a smoothstep curve over a fixed radius so delicate tips and barbs rise to full height
    dist = distance_transform_edt(mask_grid)
    u = np.clip(dist / bevel_radius_px, 0.0, 1.0)
    sculpted = 3 * (u ** 2) - 2 * (u ** 3)
    
    # Gentle broad dome over larger wing areas
    broad_dome = np.sin((dist / (dist.max() + 1e-5)) * (np.pi / 2)) * 0.25
    profile = sculpted * 0.8 + broad_dome
    
    # Gaussian filter for organic, silky smooth finish (removes faceted roughness)
    profile = gaussian_filter(profile, sigma=0.8)
    
    # Total Z height map
    Z = np.where(mask_grid, base_thickness + (profile * bevel_height), 0.0)
    
    # Pad edges with zeros so side walls seal cleanly
    Z = np.pad(Z, pad_width=1, mode="constant", constant_values=0.0)
    mask_grid = Z > 0
    
    H, W = Z.shape
    scale = width_mm / W
    
    faces = []
    print("Building watertight 3D solid mesh...")
    
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
                
                # Flat bottom base (Z=0)
                faces.append([[x0, y0, 0], [x1, y0, 0], [x0, y1, 0]])
                faces.append([[x1, y0, 0], [x1, y1, 0], [x0, y1, 0]])
                
                # Watertight vertical boundary walls
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
    print(f"Success! Saved {output_stl} ({len(faces_arr):,} triangles).")

if __name__ == "__main__":
    create_heart_stl()
