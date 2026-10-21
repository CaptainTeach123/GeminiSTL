/**
 * GeminiSTL - High-Precision Watertight 3D STL Mesh Generator
 * Features chiseled tribal blade ridge sculpting, smart reflection notch healing,
 * studio lighting and Three.js 3D viewport.
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
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) rowIn[x] = f[y * w + x];
    const rowOut = dt1d(rowIn, w);
    for (let x = 0; x < w; x++) {
      dist[y * w + x] = Math.sqrt(rowOut[x]);
    }
  }
  return dist;
}

// Separable Gaussian filter
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

// Morphological closing (iterations=5 to seal horn reflection crevices completely)
function binaryClosing(mask, w, h, iterations = 5) {
  let cur = new Uint8Array(mask);
  for (let it = 0; it < iterations; it++) {
    const dilated = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) {
      const row = y * w;
      for (let x = 0; x < w; x++) {
        if (cur[row + x]) {
          dilated[row + x] = 1;
          if (x > 0) dilated[row + x - 1] = 1;
          if (x < w - 1) dilated[row + x + 1] = 1;
          if (y > 0) dilated[row - w + x] = 1;
          if (y < h - 1) dilated[row + w + x] = 1;
        }
      }
    }
    cur = dilated;
  }
  for (let it = 0; it < iterations; it++) {
    const eroded = new Uint8Array(w * h);
    for (let y = 1; y < h - 1; y++) {
      const row = y * w;
      for (let x = 1; x < w - 1; x++) {
        if (cur[row + x] &&
            cur[row + x - 1] && cur[row + x + 1] &&
            cur[row - w + x] && cur[row + w + x]) {
          eroded[row + x] = 1;
        }
      }
    }
    cur = eroded;
  }
  return cur;
}

// Connected Component filtering: removes background debris & dark corner vignette
function filterConnectedComponents(mask, w, h, data) {
  const visited = new Uint8Array(w * h);
  const cleanMask = new Uint8Array(w * h);
  const queue = new Int32Array(w * h);
  const minSize = Math.max(60, Math.round((w * h) / 1200));

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const startIdx = y * w + x;
      if (!mask[startIdx] || visited[startIdx]) continue;

      let head = 0, tail = 0;
      queue[tail++] = startIdx;
      visited[startIdx] = 1;

      let sumG = 0, sumB = 0, sumLum = 0;

      while (head < tail) {
        const cur = queue[head++];
        const cx = cur % w;
        const cy = Math.floor(cur / w);

        const p = cur * 4;
        const r = data[p], g = data[p + 1], b = data[p + 2];
        sumG += g;
        sumB += b;
        sumLum += (0.299 * r + 0.587 * g + 0.114 * b);

        if (cx > 0) { const n = cur - 1; if (mask[n] && !visited[n]) { visited[n] = 1; queue[tail++] = n; } }
        if (cx < w - 1) { const n = cur + 1; if (mask[n] && !visited[n]) { visited[n] = 1; queue[tail++] = n; } }
        if (cy > 0) { const n = cur - w; if (mask[n] && !visited[n]) { visited[n] = 1; queue[tail++] = n; } }
        if (cy < h - 1) { const n = cur + w; if (mask[n] && !visited[n]) { visited[n] = 1; queue[tail++] = n; } }
      }

      const compSize = tail;
      const meanG = sumG / compSize;
      const meanB = sumB / compSize;
      const meanLum = sumLum / compSize;

      // Retain all genuine heart features (including delicate tribal hooks, spurs, and barbs)
      // Discards only dark background vignette (near black, lum < 30)
      if (meanLum > 30 && (meanG > 15 || meanB > 25 || compSize > 1000)) {
        for (let i = 0; i < tail; i++) {
          cleanMask[queue[i]] = 1;
        }
      }
    }
  }
  return cleanMask;
}

// Fill internal reflection voids while preserving genuine tribal negative cutouts
function fillSmallHoles(mask, w, h, maxHoleSize = 300) {
  const visited = new Uint8Array(w * h);
  const queue = new Int32Array(w * h);
  const filled = new Uint8Array(mask);

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const startIdx = y * w + x;
      if (mask[startIdx] || visited[startIdx]) continue;

      let head = 0, tail = 0;
      queue[tail++] = startIdx;
      visited[startIdx] = 1;
      let touchesBorder = false;

      while (head < tail) {
        const cur = queue[head++];
        const cx = cur % w;
        const cy = Math.floor(cur / w);

        if (cx === 0 || cx === w - 1 || cy === 0 || cy === h - 1) {
          touchesBorder = true;
        }

        if (cx > 0) { const n = cur - 1; if (!mask[n] && !visited[n]) { visited[n] = 1; queue[tail++] = n; } }
        if (cx < w - 1) { const n = cur + 1; if (!mask[n] && !visited[n]) { visited[n] = 1; queue[tail++] = n; } }
        if (cy > 0) { const n = cur - w; if (!mask[n] && !visited[n]) { visited[n] = 1; queue[tail++] = n; } }
        if (cy < h - 1) { const n = cur + w; if (!mask[n] && !visited[n]) { visited[n] = 1; queue[tail++] = n; } }
      }

      if (!touchesBorder && tail < maxHoleSize) {
        for (let i = 0; i < tail; i++) {
          filled[queue[i]] = 1;
        }
      }
    }
  }
  return filled;
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
  modelStyle: document.getElementById("modelStyle"),
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
let currentMaterialMode = "studio";
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

  // Studio Lighting (eliminates dark voids, reveals all 3D curves)
  const hemiLight = new THREE.HemisphereLight(0xffffff, 0x1e293b, 0.75);
  scene.add(hemiLight);

  const mainLight = new THREE.DirectionalLight(0xffffff, 0.95);
  mainLight.position.set(120, 140, 220);
  mainLight.castShadow = true;
  scene.add(mainLight);

  const fillLight = new THREE.DirectionalLight(0xa5b4fc, 0.5);
  fillLight.position.set(-150, -100, 120);
  scene.add(fillLight);

  const rimLight = new THREE.DirectionalLight(0xf472b6, 0.4);
  rimLight.position.set(0, 220, 80);
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
    studio: { color: 0xf1f5f9, metalness: 0.35, roughness: 0.22 },
    chrome: { color: 0xdde3ea, metalness: 0.88, roughness: 0.16 },
    pink: { color: 0xec4899, metalness: 0.15, roughness: 0.3 },
    cyan: { color: 0x06b6d4, metalness: 0.25, roughness: 0.3 },
    gold: { color: 0xf59e0b, metalness: 0.85, roughness: 0.2 },
    clay: { color: 0xa8a29e, metalness: 0.05, roughness: 0.7 }
  };
  const cfg = matConfig[currentMaterialMode] || matConfig.studio;
  return new THREE.MeshStandardMaterial({
    color: cfg.color,
    metalness: cfg.metalness,
    roughness: cfg.roughness,
    wireframe: isWireframe,
    side: THREE.DoubleSide
  });
}

// --- Image Loading & Segmentation ---
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
    setTimeout(build3DModel, 50);
  };
  img.onerror = () => {
    hideLoader();
    alert("Could not load image: " + src);
  };
  img.src = src;
}

function getMaskFromImage(img, maxSize) {
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
  const rawMask = new Uint8Array(w * h);
  const mode = dom.maskMode.value;
  const lumThresh = parseFloat(dom.lumThreshold.value) / 100;

  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    const a = data[i + 3];

    const lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;

    let isForeground = false;
    if (mode === "tribal") {
      // Rejection of red textured background (capped at r <= 205 so bright red metallic highlights remain):
      const isBg = (r > 35) && (r <= 205) && (r > 1.8 * g) && (r > 1.8 * b) && (g < 35) && (b < 75);
      isForeground = !isBg;
    } else if (mode === "bright") {
      isForeground = lum >= lumThresh && a > 30;
    } else if (mode === "dark") {
      isForeground = lum < lumThresh && a > 30;
    } else if (mode === "alpha") {
      isForeground = a > 128;
    }
    rawMask[p] = isForeground ? 1 : 0;
  }

  let finalMask = rawMask;
  let smoothFloat = null;
  if (mode === "tribal") {
    // 1. Filter out corner vignette & isolated debris on raw mask
    const filtered = filterConnectedComponents(rawMask, w, h, data);

    // Clear 3-pixel border to guarantee background connectivity around the model
    for (let x = 0; x < w; x++) {
      filtered[x] = 0; filtered[(h - 1) * w + x] = 0;
      if (h > 2) { filtered[w + x] = 0; filtered[(h - 2) * w + x] = 0; }
      if (h > 4) { filtered[2 * w + x] = 0; filtered[(h - 3) * w + x] = 0; }
    }
    for (let y = 0; y < h; y++) {
      filtered[y * w] = 0; filtered[y * w + w - 1] = 0;
      if (w > 2) { filtered[y * w + 1] = 0; filtered[y * w + w - 2] = 0; }
      if (w > 4) { filtered[y * w + 2] = 0; filtered[y * w + w - 3] = 0; }
    }

    // 2. Heal small internal specular reflection voids while keeping the 4 genuine cutouts 100% OPEN!
    const maxNoiseHole = Math.max(30, Math.round((w * h) / 2800));
    const filled = fillSmallHoles(filtered, w, h, maxNoiseHole);

    // 3. Level-set contour smoothing: turns jagged raster staircases into silky smooth CAD curves
    smoothFloat = gaussianFilter(filled, w, h, 2.5);
    finalMask = new Uint8Array(w * h);
    for (let i = 0; i < w * h; i++) {
      finalMask[i] = smoothFloat[i] >= 0.5 ? 1 : 0;
    }
  } else {
    smoothFloat = gaussianFilter(finalMask, w, h, 2.0);
  }

  return { w, h, mask: finalMask, smoothFloat, canvas };
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
      imgData.data[p] = 59;
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

// --- 3D Mesh Generation & Chiseled Ridge Profiling ---
function build3DModel() {
  if (!state.image) return;

  const startTime = performance.now();
  showLoader("Sculpting Chiseled 3D Blade Ridges...");

  setTimeout(() => {
    const maxSize = parseInt(dom.maxResolution.value, 10);
    const isDoubleSided = !dom.modelStyle || dom.modelStyle.value === "double";
    const widthMm = parseFloat(dom.widthMm.value);
    const baseThickness = parseFloat(dom.baseThickness.value);
    const bevelHeight = parseFloat(dom.bevelHeight.value);
    const sigma = parseFloat(dom.smoothingSigma.value);

    const { w, h, mask: rawMask, smoothFloat, canvas } = getMaskFromImage(state.image, maxSize);
    const ctx = canvas.getContext("2d");
    const imgData = ctx.getImageData(0, 0, w, h).data;

    // 1. Distance transform with Gaussian filter (sigma=1.5 removes discrete Voronoi grid discretization)
    const rawDist = distanceTransform(rawMask, w, h);
    const dist = gaussianFilter(rawDist, w, h, 1.5);

    // 2. Pure Geometric CAD Blade Sculpting:
    // Continuous sinusoidal chamfer bevel rising cleanly from perimeter knife-edge (d=0, Z=0)
    // Zero flat brim/flange, zero artificial feathering
    const Z = new Float32Array(w * h);
    const halfH = isDoubleSided ? ((baseThickness + bevelHeight) / 2.0) : bevelHeight;

    for (let i = 0; i < w * h; i++) {
      if (!rawMask[i]) continue;
      const d = dist[i];
      const tBevel = Math.min(d / 14.0, 1.0);
      const bevelProfile = Math.sin(tBevel * (Math.PI / 2.0));
      const bladeRoof = 1.0 - Math.exp(-d / 8.0);
      const sculpted = (bevelProfile * 0.70) + (bladeRoof * 0.30);
      if (isDoubleSided) {
        Z[i] = sculpted * halfH;
      } else {
        const baseTaper = Math.min(d / 4.0, 1.0);
        Z[i] = (baseTaper * baseThickness) + (sculpted * bevelHeight);
      }
    }

    // 3. Sub-pixel Marching Squares edge crossings
    const hCross = new Float32Array(h * (w - 1)).fill(-1);
    for (let y = 0; y < h; y++) {
      const rowOffset = y * w;
      const hOffset = y * (w - 1);
      for (let x = 0; x < w - 1; x++) {
        const v0 = smoothFloat[rowOffset + x], v1 = smoothFloat[rowOffset + x + 1];
        if ((v0 >= 0.5) !== (v1 >= 0.5)) {
          const t = Math.max(0.0001, Math.min(0.9999, (0.5 - v0) / (v1 - v0 + 1e-12)));
          hCross[hOffset + x] = x + t;
        }
      }
    }

    const vCross = new Float32Array((h - 1) * w).fill(-1);
    for (let y = 0; y < h - 1; y++) {
      const row0 = y * w, row1 = (y + 1) * w;
      for (let x = 0; x < w; x++) {
        const v0 = smoothFloat[row0 + x], v1 = smoothFloat[row1 + x];
        if ((v0 >= 0.5) !== (v1 >= 0.5)) {
          const t = Math.max(0.0001, Math.min(0.9999, (0.5 - v0) / (v1 - v0 + 1e-12)));
          vCross[row0 + x] = y + t;
        }
      }
    }

    const scale = widthMm / w;
    const cx = (w * scale) / 2;
    const cy = (h * scale) / 2;

    const topGridV = new Int32Array(w * h).fill(-1);
    const botGridV = new Int32Array(w * h).fill(-1);
    const topHCrossV = new Int32Array(h * (w - 1)).fill(-1);
    const botHCrossV = new Int32Array(h * (w - 1)).fill(-1);
    const topVCrossV = new Int32Array((h - 1) * w).fill(-1);
    const botVCrossV = new Int32Array((h - 1) * w).fill(-1);
    const positions = [];
    const indices = [];

    function getTopGrid(x, y, z) {
      const idx = y * w + x;
      if (topGridV[idx] === -1) {
        topGridV[idx] = positions.length / 3;
        positions.push(x * scale - cx, (h - 1 - y) * scale - cy, z);
      }
      return topGridV[idx];
    }

    function getBotGrid(x, y, z) {
      const idx = y * w + x;
      if (botGridV[idx] === -1) {
        botGridV[idx] = positions.length / 3;
        positions.push(x * scale - cx, (h - 1 - y) * scale - cy, isDoubleSided ? -z : 0.0);
      }
      return botGridV[idx];
    }

    function getTopHCross(x, y) {
      const idx = y * (w - 1) + x;
      if (topHCrossV[idx] === -1) {
        topHCrossV[idx] = positions.length / 3;
        positions.push(hCross[idx] * scale - cx, (h - 1 - y) * scale - cy, 0.0);
      }
      return topHCrossV[idx];
    }

    function getBotHCross(x, y) {
      const idx = y * (w - 1) + x;
      if (botHCrossV[idx] === -1) {
        botHCrossV[idx] = positions.length / 3;
        positions.push(hCross[idx] * scale - cx, (h - 1 - y) * scale - cy, 0.0);
      }
      return botHCrossV[idx];
    }

    function getTopVCross(x, y) {
      const idx = y * w + x;
      if (topVCrossV[idx] === -1) {
        topVCrossV[idx] = positions.length / 3;
        positions.push(x * scale - cx, (h - 1 - vCross[idx]) * scale - cy, 0.0);
      }
      return topVCrossV[idx];
    }

    function getBotVCross(x, y) {
      const idx = y * w + x;
      if (botVCrossV[idx] === -1) {
        botVCrossV[idx] = positions.length / 3;
        positions.push(x * scale - cx, (h - 1 - vCross[idx]) * scale - cy, 0.0);
      }
      return botVCrossV[idx];
    }

    function addTri(t0, t1, t2, b0, b1, b2) {
      indices.push(t0, t1, t2);
      indices.push(b0, b2, b1);
    }

    function addPoly(topPts, botPts) {
      for (let i = 1; i < topPts.length - 1; i++) {
        addTri(topPts[0], topPts[i], topPts[i + 1], botPts[0], botPts[i], botPts[i + 1]);
      }
    }

    for (let y = 0; y < h - 1; y++) {
      const r0 = y, r1 = y + 1;
      const row0 = r0 * w, row1 = r1 * w;
      for (let x = 0; x < w - 1; x++) {
        const c0 = x, c1 = x + 1;
        const b0 = rawMask[row0 + c0];
        const b1 = rawMask[row0 + c1];
        const b2 = rawMask[row1 + c1];
        const b3 = rawMask[row1 + c0];
        const code = b0 | (b1 << 1) | (b2 << 2) | (b3 << 3);
        if (code === 0) continue;

        const zTL = Z[row0 + c0], zTR = Z[row0 + c1], zBR = Z[row1 + c1], zBL = Z[row1 + c0];
        const TL_t = getTopGrid(c0, r0, zTL), TL_b = getBotGrid(c0, r0, zTL);
        const TR_t = getTopGrid(c1, r0, zTR), TR_b = getBotGrid(c1, r0, zTR);
        const BR_t = getTopGrid(c1, r1, zBR), BR_b = getBotGrid(c1, r1, zBR);
        const BL_t = getTopGrid(c0, r1, zBL), BL_b = getBotGrid(c0, r1, zBL);

        if (code === 15) {
          addTri(BL_t, BR_t, TR_t, BL_b, BR_b, TR_b);
          addTri(BL_t, TR_t, TL_t, BL_b, TR_b, TL_b);
          continue;
        }

        const Et_t = ((code & 3) === 1 || (code & 3) === 2) ? getTopHCross(c0, r0) : -1;
        const Et_b = ((code & 3) === 1 || (code & 3) === 2) ? getBotHCross(c0, r0) : -1;
        const Er_t = ((code & 6) === 2 || (code & 6) === 4) ? getTopVCross(c1, r0) : -1;
        const Er_b = ((code & 6) === 2 || (code & 6) === 4) ? getBotVCross(c1, r0) : -1;
        const Eb_t = ((code & 12) === 4 || (code & 12) === 8) ? getTopHCross(c0, r1) : -1;
        const Eb_b = ((code & 12) === 4 || (code & 12) === 8) ? getBotHCross(c0, r1) : -1;
        const El_t = ((code & 9) === 1 || (code & 9) === 8) ? getTopVCross(c0, r0) : -1;
        const El_b = ((code & 9) === 1 || (code & 9) === 8) ? getBotVCross(c0, r0) : -1;

        let pTop = null, pBot = null;
        if (code === 1) { pTop = [TL_t, El_t, Et_t]; pBot = [TL_b, El_b, Et_b]; }
        else if (code === 2) { pTop = [TR_t, Et_t, Er_t]; pBot = [TR_b, Et_b, Er_b]; }
        else if (code === 4) { pTop = [BR_t, Er_t, Eb_t]; pBot = [BR_b, Er_b, Eb_b]; }
        else if (code === 8) { pTop = [BL_t, Eb_t, El_t]; pBot = [BL_b, Eb_b, El_b]; }
        else if (code === 3) { pTop = [TL_t, El_t, Er_t, TR_t]; pBot = [TL_b, El_b, Er_b, TR_b]; }
        else if (code === 6) { pTop = [TR_t, Et_t, Eb_t, BR_t]; pBot = [TR_b, Et_b, Eb_b, BR_b]; }
        else if (code === 12) { pTop = [BR_t, Er_t, El_t, BL_t]; pBot = [BR_b, Er_b, El_b, BL_b]; }
        else if (code === 9) { pTop = [BL_t, Eb_t, Et_t, TL_t]; pBot = [BL_b, Eb_b, Et_b, TL_b]; }
        else if (code === 14) { pTop = [BL_t, BR_t, TR_t, Et_t, El_t]; pBot = [BL_b, BR_b, TR_b, Et_b, El_b]; }
        else if (code === 13) { pTop = [BL_t, BR_t, Er_t, Et_t, TL_t]; pBot = [BL_b, BR_b, Er_b, Et_b, TL_b]; }
        else if (code === 11) { pTop = [BL_t, Eb_t, Er_t, TR_t, TL_t]; pBot = [BL_b, Eb_b, Er_b, TR_b, TL_b]; }
        else if (code === 7) { pTop = [El_t, Eb_t, BR_t, TR_t, TL_t]; pBot = [El_b, Eb_b, BR_b, TR_b, TL_b]; }
        else if (code === 5) {
          addTri(TL_t, El_t, Et_t, TL_b, El_b, Et_b);
          addTri(BR_t, Er_t, Eb_t, BR_b, Er_b, Eb_b);
          continue;
        } else if (code === 10) {
          addTri(TR_t, Et_t, Er_t, TR_b, Et_b, Er_b);
          addTri(BL_t, Eb_t, El_t, BL_b, Eb_b, El_b);
          continue;
        }

        if (pTop) addPoly(pTop, pBot);
      }
    }

    const positionFloatArray = new Float32Array(positions);
    const indexUintArray = positions.length / 3 > 65535 ? new Uint32Array(indices) : new Uint16Array(indices);
    const numTriangles = indices.length / 3;

    // Three.js BufferGeometry with shared vertices & smooth vertex normals
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positionFloatArray, 3));
    geometry.setIndex(new THREE.BufferAttribute(indexUintArray, 1));
    geometry.computeVertexNormals();

    while (meshGroup.children.length > 0) {
      const obj = meshGroup.children[0];
      if (obj.geometry) obj.geometry.dispose();
      meshGroup.remove(obj);
    }

    const mesh = new THREE.Mesh(geometry, getActiveMaterial());
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    meshGroup.add(mesh);

    const lengthY = (h * scale).toFixed(1);
    const heightZ = (baseThickness + bevelHeight).toFixed(1);

    state.stlBlob = createBinarySTL(positionFloatArray, indexUintArray);
    dom.downloadBtn.disabled = false;

    const elapsed = Math.round(performance.now() - startTime);
    dom.hudTriangles.textContent = numTriangles.toLocaleString();
    dom.hudDims.textContent = `${widthMm.toFixed(0)} × ${lengthY} × ${heightZ} mm`;
    dom.hudTime.textContent = `${elapsed} ms`;

    hideLoader();
  }, 30);
}

// --- Binary STL Exporter (Indexed Mesh) ---
function createBinarySTL(positions, indices) {
  const numTriangles = indices.length / 3;
  const bufferSize = 84 + numTriangles * 50;
  const buffer = new ArrayBuffer(bufferSize);
  const view = new DataView(buffer);

  const headerStr = "GeminiSTL Watertight Binary Mesh Export";
  for (let i = 0; i < 80; i++) {
    view.setUint8(i, i < headerStr.length ? headerStr.charCodeAt(i) : 0);
  }

  view.setUint32(80, numTriangles, true);

  let offset = 84;
  for (let t = 0; t < numTriangles; t++) {
    const i0 = indices[t * 3], i1 = indices[t * 3 + 1], i2 = indices[t * 3 + 2];
    const ax = positions[i0 * 3], ay = positions[i0 * 3 + 1], az = positions[i0 * 3 + 2];
    const bx = positions[i1 * 3], by = positions[i1 * 3 + 1], bz = positions[i1 * 3 + 2];
    const cx = positions[i2 * 3], cy = positions[i2 * 3 + 1], cz = positions[i2 * 3 + 2];

    const abx = bx - ax, aby = by - ay, abz = bz - az;
    const acx = cx - ax, acy = cy - ay, acz = cz - az;
    let nx = aby * acz - abz * acy;
    let ny = abz * acx - abx * acz;
    let nz = abx * acy - aby * acx;
    const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
    if (len > 0) { nx /= len; ny /= len; nz /= len; }

    view.setFloat32(offset, nx, true); offset += 4;
    view.setFloat32(offset, ny, true); offset += 4;
    view.setFloat32(offset, nz, true); offset += 4;

    view.setFloat32(offset, ax, true); offset += 4;
    view.setFloat32(offset, ay, true); offset += 4;
    view.setFloat32(offset, az, true); offset += 4;

    view.setFloat32(offset, bx, true); offset += 4;
    view.setFloat32(offset, by, true); offset += 4;
    view.setFloat32(offset, bz, true); offset += 4;

    view.setFloat32(offset, cx, true); offset += 4;
    view.setFloat32(offset, cy, true); offset += 4;
    view.setFloat32(offset, cz, true); offset += 4;

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

  dom.maskMode.addEventListener("change", e => {
    const val = e.target.value;
    dom.customMaskControls.style.display = (val === "bright" || val === "dark") ? "block" : "none";
    updateMaskPreview();
  });

  dom.generateBtn.addEventListener("click", build3DModel);
  dom.downloadBtn.addEventListener("click", downloadSTL);

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

  dom.loadPresetBtn.addEventListener("click", () => {
    dom.maskMode.value = "tribal";
    dom.customMaskControls.style.display = "none";
    loadImage("high_res.png", "tribal_heart.stl");
  });

  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(btn.dataset.tab).classList.add("active");
    });
  });

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

window.addEventListener("DOMContentLoaded", () => {
  init3D();
  setupEvents();
  loadImage("high_res.png", "tribal_heart.stl");
});
