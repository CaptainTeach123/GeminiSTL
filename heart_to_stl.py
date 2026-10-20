import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter, label, binary_closing, generate_binary_structure
from stl import mesh

def create_heart_stl(
    image_path="high_res.png",
    output_stl="tribal_heart.stl",
    max_size=350,          # Grid resolution (higher = smoother edges, larger file)
    width_mm=120.0,        # Physical width of the 3D print in millimeters
    base_thickness=3.0,    # Solid base thickness in mm
    bevel_height=4.5,      # Additional sculpted bevel height in mm
):
    print(f"Loading {image_path}...")
    img = Image.open(image_path).convert("RGB")
    
    # Resize while keeping aspect ratio for clean, fast mesh generation
    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    
    R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    
    # 1. Advanced background separation:
    # The textured background has high Red, but very low Green (G <= 30) and low Blue.
    # Background satisfies: (R > 35) & (R > 1.8 * G) & (R > 1.8 * B) & (G < 35)
    is_bg = (R > 35) & (R > 1.8 * G) & (R > 1.8 * B) & (G < 35)
    raw_mask = ~is_bg
    
    # 2. Morphological closing to seal fine creases/reflection borders inside chrome facets
    struct = generate_binary_structure(2, 2)
    closed_mask = binary_closing(raw_mask, structure=struct, iterations=1)
    
    # 3. Connected component filtering: eliminates background grunge, floating speckles, and corner vignettes
    labeled, num_features = label(closed_mask)
    sizes = np.bincount(labeled.ravel())
    
    border_labels = set(labeled[0, :]).union(labeled[-1, :]).union(labeled[:, 0]).union(labeled[:, -1])
    
    mask = np.zeros_like(closed_mask, dtype=bool)
    for lbl in range(1, num_features + 1):
        sz = sizes[lbl]
        if sz < 80 or lbl in border_labels:
            continue
        comp_pixels = arr[labeled == lbl]
        mean_g = comp_pixels[:, 1].mean()
        mean_lum = (0.299 * comp_pixels[:, 0] + 0.587 * comp_pixels[:, 1] + 0.114 * comp_pixels[:, 2]).mean()
        # Keep bright chrome / metallic components, discard dark vignette/noise
        if mean_g > 20 and mean_lum > 35:
            mask |= (labeled == lbl)
            
    print(f"Isolated clean foreground heart mesh ({mask.sum()} grid cells)...")
    
    # Compute distance from the outer edges to create a sculpted 3D bevel
    dist = distance_transform_edt(mask)
    if dist.max() > 0:
        dist = dist / dist.max()
    
    # Blend distance bevel with surface shading for extra detail
    luminance = (0.299 * R + 0.587 * G + 0.114 * B) / 255.0
    height_profile = (dist * 0.75) + (luminance * 0.25)
    height_profile = gaussian_filter(height_profile, sigma=1.0)
    
    # Total Z height map
    Z = np.where(mask, base_thickness + (height_profile * bevel_height), 0.0)
    
    # Pad edges with zeros so the outer boundary closes cleanly
    Z = np.pad(Z, pad_width=1, mode="constant", constant_values=0.0)
    mask = Z > 0
    
    H, W = Z.shape
    scale = width_mm / W
    
    faces = []
    print("Building 3D watertight geometry...")
    
    for r in range(H - 1):
        for c in range(W - 1):
            if mask[r, c] or mask[r+1, c] or mask[r, c+1] or mask[r+1, c+1]:
                x0, x1 = c * scale, (c + 1) * scale
                # Flip Y so the heart isn't upside down
                y0, y1 = (H - r) * scale, (H - (r + 1)) * scale
                
                z00, z10 = Z[r, c], Z[r, c+1]
                z01, z11 = Z[r+1, c], Z[r+1, c+1]
                
                # Top sculpted surface (2 triangles)
                faces.append([[x0, y0, z00], [x0, y1, z01], [x1, y0, z10]])
                faces.append([[x1, y0, z10], [x0, y1, z01], [x1, y1, z11]])
                
                # Flat bottom surface at Z=0 (2 triangles, reversed winding)
                faces.append([[x0, y0, 0], [x1, y0, 0], [x0, y1, 0]])
                faces.append([[x1, y0, 0], [x1, y1, 0], [x0, y1, 0]])
                
                # Side walls wherever a pixel transitions to empty background
                if r == 0 or not mask[r-1, c]:
                    faces.append([[x0, y0, 0], [x0, y0, z00], [x1, y0, z10]])
                    faces.append([[x0, y0, 0], [x1, y0, z10], [x1, y0, 0]])
                if r == H - 2 or not mask[r+1, c]:
                    faces.append([[x0, y1, 0], [x1, y1, z11], [x0, y1, z01]])
                    faces.append([[x0, y1, 0], [x1, y1, 0], [x1, y1, z11]])
                if c == 0 or not mask[r, c-1]:
                    faces.append([[x0, y0, 0], [x0, y1, z01], [x0, y0, z00]])
                    faces.append([[x0, y0, 0], [x0, y1, 0], [x0, y1, z01]])
                if c == W - 2 or not mask[r, c+1]:
                    faces.append([[x1, y0, 0], [x1, y0, z10], [x1, y1, z11]])
                    faces.append([[x1, y0, 0], [x1, y1, z11], [x1, y1, 0]])

    faces_arr = np.array(faces, dtype=np.float32)
    heart_mesh = mesh.Mesh(np.zeros(faces_arr.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces_arr):
        heart_mesh.vectors[i] = f
        
    heart_mesh.save(output_stl)
    print(f"Success! Saved {output_stl} ({len(faces_arr)} triangles).")

if __name__ == "__main__":
    create_heart_stl()
