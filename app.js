/**
 * GeminiSTL - 2D Image to Watertight 3D STL Mesh Generator
 * Features exact Euclidean Distance Transform (EDT), Gaussian smoothing,
 * watertight base/wall closure, Three.js 3D viewport, and binary STL export.
 */

// --- Euclidean Distance Transform (Felzenszwalb & Huttenlocher 1D/2D) ---
function dt1d(f, n) {
  const d = new Float32Array(n);
  const v = new Int32Array(n);
  const z = new Float32Array(n + 1);
  let k = 0;
  v[0] = 0;
  z[0] = -1e20;
  z[1] = 1e20;

  for (let q = 1; q < n; q++) {
    let s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
    while (s <= z[k]) {
      k--;
      s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
    }
    k++;
    v[k] = q;
    z[k] = s;
    z[k + 1] = 1e20;
  }

  k = 0;
  for (let q = 0; q < n; q++) {
    while (z[k + 1] < q) k++;
    d[q] = (q - v[k]) * (q - v[k]) + f[v[k]];
  }
  return d;
}

// Compute Euclidean distance from mask foreground pixels to the nearest background pixel (0)
function distanceTransform(mask, w, h) {
  const INF = 1e9;
  let f = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    f[i] = mask[i] ? INF : 0;
  }

  // Column pass
  const colIn = new Float32Array(h);
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) colIn[y] = f[y * w + x];
    const colOut = dt1d(colIn, h);
    for (let y = 0; y < h; y++) f[y * w + x] = colOut[y];
  }

  // Row pass
  const rowIn = new Float32Array(w);
  const dist = new Float32Array(w * h);
  let maxD = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) rowIn[x] = f[y * w + x];
    const rowOut = dt1d(rowIn, w);
    for (let x = 0; x < w; x++) {
      const d = Math.sqrt(rowOut[x]);
      dist[y * w + x] = d;
      if (d > maxD) maxD = d;
    }
  }

  // Normalize to [0, 1] if maxD > 0
  if (maxD > 0) {
    for (let i = 0; i < w * h; i++) dist[i] /= maxD;
  }
  return dist;
}

// Separable 1D Gaussian filter
function gaussianFilter(arr, w, h, sigma) {
  if (sigma <= 0) return arr.slice();
  const radius = Math.ceil(sigma * 3);
  const kernelSize = 2 * radius + 1;
  const kernel = new Float32Array(kernelSize);
  let sum = 0;
  for (let i = -radius; i <= radius; i++) {
    const val = Math.exp(-(i * i) / (2 * sigma * sigma));
    kernel[i + radius] = val;
    sum += val;
  }
  for (let i = 0; i < kernelSize; i++) kernel[i] /= sum;

  const temp = new Float32Array(w * h);
  // Horizontal pass
  for (let y = 0; y < h; y++) {
    const rowOffset = y * w;
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let k = -radius; k <= radius; k++) {
        const nx = Math.min(w - 1, Math.max(0, x + k));
        acc += arr[rowOffset + nx] * kernel[k + radius];
      }
      temp[rowOffset + x] = acc;
    }
  }

  const out = new Float32Array(w * h);
  // Vertical pass
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) {
      let acc = 0;
      for (let k = -radius; k <= radius; k++) {
        const ny = Math.min(h - 1, Math.max(0, y + k));
        acc += temp[ny * w + x] * kernel[k + radius];
      }
      out[y * w + x] = acc;
    }
  }
  return out;
}

// --- App State & DOM Elements ---
const state = {
  image: null,
  imageSrc: "high_res.png",
  generatedMesh: null,
  stlBlob: null,
  stlFilename: "tribal_heart.stl"
};

const dom = {
  fileInput: document.getElementById("fileInput"),
  dropZone: document.getElementById("dropZone"),
  loadPresetBtn: document.getElementById("loadPresetBtn"),
  sourcePreview: document.getElementById("sourcePreview"),
  maskCanvas: document.getElementById("maskCanvas"),
  widthMm: document.getElementById("widthMm"),
  widthMmVal: document.getElementById("widthMmVal"),
  baseThickness: document.getElementById("baseThickness"),
  baseThicknessVal: document.getElementById("baseThicknessVal"),
  bevelHeight: document.getElementById("bevelHeight"),
  bevelHeightVal: document.getElementById("bevelHeightVal"),
  maxResolution: document.getElementById("maxResolution"),
  maxResolutionVal: document.getElementById("maxResolutionVal"),
  smoothingSigma: document.getElementById("smoothingSigma"),
  smoothingSigmaVal: document.getElementById("smoothingSigmaVal"),
  maskMode: document.getElementById("maskMode"),
  customMaskControls: document.getElementById("customMaskControls"),
  lumThreshold: document.getElementById("lumThreshold"),
  lumThresholdVal: document.getElementById("lumThresholdVal"),
  generateBtn: document.getElementById("generateBtn"),
  downloadBtn: document.getElementById("downloadBtn"),
  viewport3d: document.getElementById("viewport3d"),
  hudTriangles: document.getElementById("hudTriangles"),
  hudDims: document.getElementById("hudDims"),
  hudTime: document.getElementById("hudTime"),
  resetCamBtn: document.getElementById("resetCamBtn"),
  wireframeBtn: document.getElementById("wireframeBtn"),
  materialSelect: document.getElementById("materialSelect"),
  loaderOverlay: document.getElementById("loaderOverlay"),
  loaderStatus: document.getElementById("loaderStatus")
};

// --- Three.js 3D Viewer Setup ---
let scene, camera, renderer, controls, meshGroup;
let currentMaterialMode = "chrome";
let isWireframe = false;

function init3D() {
  const container = dom.viewport3d;
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0c1017);

  camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 1, 2000);
  camera.position.set(0, -180, 160);

  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  container.appendChild(renderer.domElement);

  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.maxDistance = 1000;
  controls.minDistance = 20;

  // Lighting
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.45);
  scene.add(ambientLight);

  const mainLight = new THREE.DirectionalLight(0xffffff, 0.9);
  mainLight.position.set(100, 120, 200);
  mainLight.castShadow = true;
  scene.add(mainLight);

  const fillLight = new THREE.DirectionalLight(0x90b0ff, 0.5);
  fillLight.position.set(-150, -100, 100);
  scene.add(fillLight);

  const rimLight = new THREE.DirectionalLight(0xec4899, 0.4);
  rimLight.position.set(0, 200, 50);
  scene.add(rimLight);

  // Bed Grid Floor
  const gridHelper = new THREE.GridHelper(250, 25, 0x3b82f6, 0x1e293b);
  gridHelper.rotation.x = Math.PI / 2;
  gridHelper.position.z = -0.05;
  scene.add(gridHelper);

  meshGroup = new THREE.Group();
  scene.add(meshGroup);

  window.addEventListener("resize", onWindowResize);
  animate();
}

function onWindowResize() {
  const container = dom.viewport3d;
  if (!container || !renderer) return;
  camera.aspect = container.clientWidth / container.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(container.clientWidth, container.clientHeight);
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

function getActiveMaterial() {
  const matConfig = {
    chrome: { color: 0xd8dde6, metalness: 0.88, roughness: 0.15, clearcoat: 0.3 },
    pink: { color: 0xec4899, metalness: 0.1, roughness: 0.4 },
    cyan: { color: 0x06b6d4, metalness: 0.2, roughness: 0.35 },
    gold: { color: 0xf59e0b, metalness: 0.8, roughness: 0.25 },
    clay: { color: 0xa8a29e, metalness: 0.0, roughness: 0.85 }
  };
  const cfg = matConfig[currentMaterialMode] || matConfig.chrome;
  return new THREE.MeshStandardMaterial({
    color: cfg.color,
    metalness: cfg.metalness,
    roughness: cfg.roughness,
    wireframe: isWireframe,
    side: THREE.DoubleSide
  });
}

// --- Image Loading & Mask Processing ---
function loadImage(src, name) {
  showLoader("Loading source image...");
  const img = new Image();
  img.crossOrigin = "anonymous";
  img.onload = () => {
    state.image = img;
    state.stlFilename = name ? name.replace(/\.[^/.]+$/, "") + ".stl" : "model.stl";
    dom.sourcePreview.src = img.src;
    updateMaskPreview();
    hideLoader();
    // Auto-generate on first load
    setTimeout(build3DModel, 50);
  };
  img.onerror = () => {
    hideLoader();
    alert("Could not load image: " + src);
  };
  img.src = src;
}

function getMaskFromImage(img, maxSize) {
  // Resize keeping aspect ratio
  let w = img.naturalWidth || img.width;
  let h = img.naturalHeight || img.height;
  if (w > maxSize || h > maxSize) {
    if (w >= h) {
      h = Math.round((h * maxSize) / w);
      w = maxSize;
    } else {
      w = Math.round((w * maxSize) / h);
      h = maxSize;
    }
  }

  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(img, 0, 0, w, h);

  const imgData = ctx.getImageData(0, 0, w, h);
  const data = imgData.data;
  const mask = new Uint8Array(w * h);
  const luminance = new Float32Array(w * h);
  const mode = dom.maskMode.value;
  const lumThresh = parseFloat(dom.lumThreshold.value) / 100;

  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    const a = data[i + 3];

    const lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;
    luminance[p] = lum;

    let isForeground = false;
    if (mode === "tribal") {
      // Chrome/silver and neon pink edges: higher G/B or bright luminance
      isForeground = (g > 38) || (b > 45) || ((r > 210) && (g > 20));
    } else if (mode === "bright") {
      isForeground = lum >= lumThresh && a > 30;
    } else if (mode === "dark") {
      isForeground = lum < lumThresh && a > 30;
    } else if (mode === "alpha") {
      isForeground = a > 128;
    }
    mask[p] = isForeground ? 1 : 0;
  }

  return { w, h, mask, luminance, canvas };
}

function updateMaskPreview() {
  if (!state.image) return;
  const { w, h, mask } = getMaskFromImage(state.image, 250);
  const canvas = dom.maskCanvas;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  const imgData = ctx.createImageData(w, h);
  for (let i = 0, p = 0; i < mask.length; i++, p += 4) {
    if (mask[i]) {
      imgData.data[p] = 59;     // Blue accent
      imgData.data[p + 1] = 130;
      imgData.data[p + 2] = 246;
      imgData.data[p + 3] = 255;
    } else {
      imgData.data[p] = 18;
      imgData.data[p + 1] = 24;
      imgData.data[p + 2] = 36;
      imgData.data[p + 3] = 255;
    }
  }
  ctx.putImageData(imgData, 0, 0);
}

// --- 3D Mesh Generation & Watertight Closure ---
function build3DModel() {
  if (!state.image) return;

  const startTime = performance.now();
  showLoader("Generating 3D Watertight Geometry...");

  setTimeout(() => {
    const maxSize = parseInt(dom.maxResolution.value, 10);
    const widthMm = parseFloat(dom.widthMm.value);
    const baseThickness = parseFloat(dom.baseThickness.value);
    const bevelHeight = parseFloat(dom.bevelHeight.value);
    const sigma = parseFloat(dom.smoothingSigma.value);

    const { w, h, mask: rawMask, luminance } = getMaskFromImage(state.image, maxSize);

    // Compute Distance Transform
    const dist = distanceTransform(rawMask, w, h);

    // Blend distance bevel (75%) with surface luminance shading (25%)
    let heightProfile = new Float32Array(w * h);
    for (let i = 0; i < w * h; i++) {
      heightProfile[i] = (dist[i] * 0.75) + (luminance[i] * 0.25);
    }
    heightProfile = gaussianFilter(heightProfile, w, h, sigma);

    // Build padded Z grid: pad width = 1 so boundaries close cleanly
    const paddedW = w + 2;
    const paddedH = h + 2;
    const Z = new Float32Array(paddedW * paddedH);
    const mask = new Uint8Array(paddedW * paddedH);

    for (let r = 0; r < h; r++) {
      for (let c = 0; c < w; c++) {
        const rawIdx = r * w + c;
        const padIdx = (r + 1) * paddedW + (c + 1);
        if (rawMask[rawIdx]) {
          mask[padIdx] = 1;
          Z[padIdx] = baseThickness + (heightProfile[rawIdx] * bevelHeight);
        }
      }
    }

    const scale = widthMm / paddedW;
    const vertices = [];

    // Helper to push triangle
    function addTri(ax, ay, az, bx, by, bz, cx, cy, cz) {
      vertices.push(ax, ay, az, bx, by, bz, cx, cy, cz);
    }

    for (let r = 0; r < paddedH - 1; r++) {
      for (let c = 0; c < paddedW - 1; c++) {
        const p00 = r * paddedW + c;
        const p10 = r * paddedW + (c + 1);
        const p01 = (r + 1) * paddedW + c;
        const p11 = (r + 1) * paddedW + (c + 1);

        if (mask[p00] || mask[p10] || mask[p01] || mask[p11]) {
          const x0 = c * scale;
          const x1 = (c + 1) * scale;
          // Flip Y to match 3D space orientation
          const y0 = (paddedH - r) * scale;
          const y1 = (paddedH - (r + 1)) * scale;

          const z00 = Z[p00], z10 = Z[p10];
          const z01 = Z[p01], z11 = Z[p11];

          // Top sculpted surface (2 triangles)
          addTri(x0, y0, z00,  x0, y1, z01,  x1, y0, z10);
          addTri(x1, y0, z10,  x0, y1, z01,  x1, y1, z11);

          // Flat bottom surface at Z=0 (2 triangles, reversed winding)
          addTri(x0, y0, 0,  x1, y0, 0,  x0, y1, 0);
          addTri(x1, y0, 0,  x1, y1, 0,  x0, y1, 0);

          // Side walls wherever a cell transitions to background
          // North boundary
          if (r === 0 || !mask[(r - 1) * paddedW + c]) {
            addTri(x0, y0, 0,  x0, y0, z00,  x1, y0, z10);
            addTri(x0, y0, 0,  x1, y0, z10,  x1, y0, 0);
          }
          // South boundary
          if (r === paddedH - 2 || !mask[(r + 1) * paddedW + c]) {
            addTri(x0, y1, 0,  x1, y1, z11,  x0, y1, z01);
            addTri(x0, y1, 0,  x1, y1, 0,    x1, y1, z11);
          }
          // West boundary
          if (c === 0 || !mask[r * paddedW + (c - 1)]) {
            addTri(x0, y0, 0,  x0, y1, z01,  x0, y0, z00);
            addTri(x0, y0, 0,  x0, y1, 0,    x0, y1, z01);
          }
          // East boundary
          if (c === paddedW - 2 || !mask[r * paddedW + (c + 1)]) {
            addTri(x1, y0, 0,  x1, y0, z10,  x1, y1, z11);
            addTri(x1, y0, 0,  x1, y1, z11,  x1, y1, 0);
          }
        }
      }
    }

    const vertexFloatArray = new Float32Array(vertices);
    const numTriangles = vertices.length / 9;

    // Center mesh around origin on bed
    const cx = (paddedW * scale) / 2;
    const cy = (paddedH * scale) / 2;
    for (let i = 0; i < vertexFloatArray.length; i += 3) {
      vertexFloatArray[i] -= cx;
      vertexFloatArray[i + 1] -= cy;
    }

    // Build Three.js Geometry
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(vertexFloatArray, 3));
    geometry.computeVertexNormals();

    // Clear old mesh
    while (meshGroup.children.length > 0) {
      const obj = meshGroup.children[0];
      if (obj.geometry) obj.geometry.dispose();
      meshGroup.remove(obj);
    }

    const mesh = new THREE.Mesh(geometry, getActiveMaterial());
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    meshGroup.add(mesh);

    // Compute dimensions
    const lengthY = (paddedH * scale).toFixed(1);
    const heightZ = (baseThickness + bevelHeight).toFixed(1);

    // Generate Binary STL
    state.stlBlob = createBinarySTL(vertexFloatArray, numTriangles);
    dom.downloadBtn.disabled = false;

    // Update HUD
    const elapsed = Math.round(performance.now() - startTime);
    dom.hudTriangles.textContent = numTriangles.toLocaleString();
    dom.hudDims.textContent = `${widthMm.toFixed(0)} × ${lengthY} × ${heightZ} mm`;
    dom.hudTime.textContent = `${elapsed} ms`;

    hideLoader();
  }, 30);
}

// --- Binary STL Exporter ---
function createBinarySTL(vertices, numTriangles) {
  const bufferSize = 84 + numTriangles * 50;
  const buffer = new ArrayBuffer(bufferSize);
  const view = new DataView(buffer);

  // 80-byte header
  const headerStr = "GeminiSTL Watertight Binary Mesh Export";
  for (let i = 0; i < 80; i++) {
    view.setUint8(i, i < headerStr.length ? headerStr.charCodeAt(i) : 0);
  }

  // Number of triangles (uint32)
  view.setUint32(80, numTriangles, true);

  let offset = 84;
  for (let t = 0; t < numTriangles; t++) {
    const vIdx = t * 9;
    const ax = vertices[vIdx],     ay = vertices[vIdx + 1], az = vertices[vIdx + 2];
    const bx = vertices[vIdx + 3], by = vertices[vIdx + 4], bz = vertices[vIdx + 5];
    const cx = vertices[vIdx + 6], cy = vertices[vIdx + 7], cz = vertices[vIdx + 8];

    // Compute facet normal: (B - A) x (C - A)
    const abx = bx - ax, aby = by - ay, abz = bz - az;
    const acx = cx - ax, acy = cy - ay, acz = cz - az;
    let nx = aby * acz - abz * acy;
    let ny = abz * acx - abx * acz;
    let nz = abx * acy - aby * acx;
    const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
    if (len > 0) { nx /= len; ny /= len; nz /= len; }

    // Normal
    view.setFloat32(offset, nx, true); offset += 4;
    view.setFloat32(offset, ny, true); offset += 4;
    view.setFloat32(offset, nz, true); offset += 4;

    // Vertices A, B, C
    view.setFloat32(offset, ax, true); offset += 4;
    view.setFloat32(offset, ay, true); offset += 4;
    view.setFloat32(offset, az, true); offset += 4;

    view.setFloat32(offset, bx, true); offset += 4;
    view.setFloat32(offset, by, true); offset += 4;
    view.setFloat32(offset, bz, true); offset += 4;

    view.setFloat32(offset, cx, true); offset += 4;
    view.setFloat32(offset, cy, true); offset += 4;
    view.setFloat32(offset, cz, true); offset += 4;

    // 2-byte attribute byte count
    view.setUint16(offset, 0, true); offset += 2;
  }

  return new Blob([buffer], { type: "application/octet-stream" });
}

function downloadSTL() {
  if (!state.stlBlob) return;
  const url = URL.createObjectURL(state.stlBlob);
  const a = document.createElement("a");
  a.href = url;
  a.download = state.stlFilename || "tribal_heart.stl";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}

// --- UI Event Listeners ---
function setupEvents() {
  // Slider displays
  dom.widthMm.addEventListener("input", e => {
    dom.widthMmVal.textContent = `${e.target.value} mm`;
  });
  dom.baseThickness.addEventListener("input", e => {
    dom.baseThicknessVal.textContent = `${parseFloat(e.target.value).toFixed(1)} mm`;
  });
  dom.bevelHeight.addEventListener("input", e => {
    dom.bevelHeightVal.textContent = `${parseFloat(e.target.value).toFixed(1)} mm`;
  });
  dom.maxResolution.addEventListener("input", e => {
    dom.maxResolutionVal.textContent = `${e.target.value} px`;
  });
  dom.smoothingSigma.addEventListener("input", e => {
    dom.smoothingSigmaVal.textContent = parseFloat(e.target.value).toFixed(1);
  });
  dom.lumThreshold.addEventListener("input", e => {
    dom.lumThresholdVal.textContent = `${e.target.value}%`;
    updateMaskPreview();
  });

  // Mask mode selector
  dom.maskMode.addEventListener("change", e => {
    const val = e.target.value;
    dom.customMaskControls.style.display = (val === "bright" || val === "dark") ? "block" : "none";
    updateMaskPreview();
  });

  // Generate and Download buttons
  dom.generateBtn.addEventListener("click", build3DModel);
  dom.downloadBtn.addEventListener("click", downloadSTL);

  // File Upload
  dom.dropZone.addEventListener("click", () => dom.fileInput.click());
  dom.fileInput.addEventListener("change", e => {
    const file = e.target.files[0];
    if (file) handleImageFile(file);
  });

  dom.dropZone.addEventListener("dragover", e => {
    e.preventDefault();
    dom.dropZone.classList.add("dragover");
  });
  dom.dropZone.addEventListener("dragleave", () => dom.dropZone.classList.remove("dragover"));
  dom.dropZone.addEventListener("drop", e => {
    e.preventDefault();
    dom.dropZone.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) handleImageFile(file);
  });

  // Load Preset
  dom.loadPresetBtn.addEventListener("click", () => {
    dom.maskMode.value = "tribal";
    dom.customMaskControls.style.display = "none";
    loadImage("high_res.png", "tribal_heart.stl");
  });

  // Preview Tabs
  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(btn.dataset.tab).classList.add("active");
    });
  });

  // Viewport Toolbar Controls
  dom.resetCamBtn.addEventListener("click", () => {
    camera.position.set(0, -180, 160);
    controls.target.set(0, 0, 0);
  });

  dom.wireframeBtn.addEventListener("click", () => {
    isWireframe = !isWireframe;
    dom.wireframeBtn.classList.toggle("active", isWireframe);
    if (meshGroup.children[0]) {
      meshGroup.children[0].material = getActiveMaterial();
    }
  });

  dom.materialSelect.addEventListener("change", e => {
    currentMaterialMode = e.target.value;
    if (meshGroup.children[0]) {
      meshGroup.children[0].material = getActiveMaterial();
    }
  });
}

function handleImageFile(file) {
  const reader = new FileReader();
  reader.onload = e => loadImage(e.target.result, file.name);
  reader.readAsDataURL(file);
}

function showLoader(text) {
  dom.loaderStatus.textContent = text || "Processing...";
  dom.loaderOverlay.classList.remove("hidden");
}

function hideLoader() {
  dom.loaderOverlay.classList.add("hidden");
}

// Initialize on page load
window.addEventListener("DOMContentLoaded", () => {
  init3D();
  setupEvents();
  loadImage("high_res.png", "tribal_heart.stl");
});

