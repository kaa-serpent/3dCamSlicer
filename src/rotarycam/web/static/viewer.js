import * as THREE from './vendor/three.module.js';
import { OrbitControls } from './vendor/OrbitControls.js';
import { STLLoader } from './vendor/STLLoader.js';

const LAYER_LABELS = new Map([
  ['target', 'Target'],
  ['stock', 'Stock'],
  ['supports', 'Supports'],
  ['toolpaths', 'Toolpaths'],
  ['residual', 'Residual stock'],
]);

const finiteNumber = (value) => typeof value === 'number' && Number.isFinite(value);

function element(tagName, className, text) {
  const node = document.createElement(tagName);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}

class RotaryCamViewer {
  constructor(container) {
    this.container = container;
    this.manifestUrl = container.dataset.manifestUrl || '';
    this.supportPickUrl = container.dataset.supportPickUrl || '';
    this.loadVersion = 0;
    this.abortController = null;
    this.layerGroups = new Map();
    this.layerVisibility = new Map();
    this.toolVisibility = new Map();
    this.toolpathObjects = [];
    this.targetObjects = [];
    this.pointerStart = null;

    if (getComputedStyle(container).position === 'static') {
      container.style.position = 'relative';
    }
    container.setAttribute('aria-label', 'Interactive RotaryCAM 3D preview');
    container.style.overflow = 'hidden';

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x08111f);
    this.camera = new THREE.PerspectiveCamera(42, 1, 0.01, 100000);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(120, -120, 90);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.domElement.style.display = 'block';
    this.renderer.domElement.style.width = '100%';
    this.renderer.domElement.style.height = '100%';
    container.prepend(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.screenSpacePanning = true;
    this.controls.target.set(0, 0, 0);

    this.scene.add(new THREE.HemisphereLight(0xdcecff, 0x152338, 2.2));
    const keyLight = new THREE.DirectionalLight(0xffffff, 2.5);
    keyLight.position.set(-1, -2, 4);
    this.scene.add(keyLight);
    this.scene.add(new THREE.AxesHelper(30));

    for (const layer of LAYER_LABELS.keys()) {
      const group = new THREE.Group();
      group.name = `rotarycam-${layer}`;
      this.layerGroups.set(layer, group);
      this.layerVisibility.set(layer, true);
      this.scene.add(group);
    }

    this.createOverlay();
    this.installEvents();
    this.resize();
    this.animate();
  }

  createOverlay() {
    const scope = this.container.closest('.scene-column') || document;
    const cameraButtons = scope.querySelectorAll('[data-view-axis]');
    for (const button of cameraButtons) {
      const axis = button.dataset.viewAxis?.toUpperCase();
      button.addEventListener('click', () => {
        if (axis === 'ISO') {
          this.camera.up.set(0, 0, 1);
          this.fitCamera(new THREE.Vector3(1, -1, 0.7));
        } else if (['X', 'Y', 'Z'].includes(axis)) {
          this.orient(axis);
        }
      });
    }
    scope.querySelector('[data-view-fit]')?.addEventListener('click', () => this.fitCamera());

    const layerInputs = scope.querySelectorAll('[data-scene-layer]');
    for (const input of layerInputs) {
      const layer = input.dataset.sceneLayer;
      if (LAYER_LABELS.has(layer)) {
        this.layerVisibility.set(layer, input.checked);
        input.addEventListener('change', () => {
          this.layerVisibility.set(layer, input.checked);
          this.updateVisibility();
        });
      }
    }
    this.toolSelect = scope.querySelector('[data-tool-filter]');
    this.toolSelect?.addEventListener('change', () => {
      const selected = this.toolSelect.value;
      for (const toolNumber of this.toolVisibility.keys()) {
        this.toolVisibility.set(toolNumber, selected === 'all' || selected === String(toolNumber));
      }
      this.updateVisibility();
    });

    this.toolFilterRoot = null;
    if (layerInputs.length === 0) {
      this.filters = element('div', 'scene-filters');
      Object.assign(this.filters.style, {
        position: 'absolute', top: '12px', left: '12px', zIndex: '2',
        display: 'grid', gap: '5px', padding: '9px', borderRadius: '8px',
        background: 'rgba(5, 12, 24, 0.82)', color: '#dbeafe',
        font: '12px/1.3 system-ui, sans-serif', pointerEvents: 'auto',
      });
      for (const [layer, label] of LAYER_LABELS) {
        this.filters.append(this.checkbox(label, true, (visible) => {
          this.layerVisibility.set(layer, visible);
          this.updateVisibility();
        }));
      }
      this.toolFilterRoot = element('div', 'scene-tool-filters');
      this.filters.append(this.toolFilterRoot);
      this.container.append(this.filters);
    }

    this.status = element('div', 'scene-status', 'No scene generated yet.');
    this.status.setAttribute('role', 'status');
    Object.assign(this.status.style, {
      position: 'absolute', left: '50%', top: '50%', zIndex: '2',
      transform: 'translate(-50%, -50%)', maxWidth: '70%', padding: '10px 14px',
      borderRadius: '8px', background: 'rgba(5, 12, 24, 0.88)', color: '#cbd5e1',
      textAlign: 'center', font: '13px/1.4 system-ui, sans-serif', pointerEvents: 'none',
    });
    this.container.append(this.status);
    this.emptyState = this.container.querySelector('.scene-empty');
  }

  checkbox(label, checked, changed) {
    const row = element('label', 'scene-filter');
    row.style.display = 'block';
    const input = document.createElement('input');
    input.type = 'checkbox';
    input.checked = checked;
    input.addEventListener('change', () => changed(input.checked));
    row.append(input, document.createTextNode(` ${label}`));
    return row;
  }

  installEvents() {
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(this.container);
    this.renderer.domElement.addEventListener('pointerdown', (event) => {
      this.pointerStart = { x: event.clientX, y: event.clientY };
    });
    this.renderer.domElement.addEventListener('pointerup', (event) => {
      if (!this.pointerStart) return;
      const distance = Math.hypot(
        event.clientX - this.pointerStart.x,
        event.clientY - this.pointerStart.y,
      );
      this.pointerStart = null;
      if (distance <= 4 && event.button === 0) this.pickTarget(event);
    });
  }

  animate() {
    this.animationFrame = requestAnimationFrame(() => this.animate());
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  resize() {
    const width = Math.max(this.container.clientWidth, 1);
    const height = Math.max(this.container.clientHeight, 1);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height, false);
  }

  setStatus(message, kind = 'info') {
    this.status.textContent = message;
    this.status.dataset.kind = kind;
    this.status.style.display = message ? 'block' : 'none';
    this.status.style.color = kind === 'error' ? '#fecaca' : '#cbd5e1';
  }

  clearGeometry() {
    this.targetObjects = [];
    this.toolpathObjects = [];
    this.toolVisibility.clear();
    this.toolFilterRoot?.replaceChildren();
    for (const group of this.layerGroups.values()) {
      for (const child of [...group.children]) {
        group.remove(child);
        child.geometry?.dispose();
        if (Array.isArray(child.material)) {
          for (const material of child.material) material.dispose();
        } else {
          child.material?.dispose();
        }
      }
    }
  }

  async load() {
    this.manifestUrl = this.container.dataset.manifestUrl || this.manifestUrl;
    this.supportPickUrl = this.container.dataset.supportPickUrl || this.supportPickUrl;
    if (!this.manifestUrl) {
      this.clearGeometry();
      this.setStatus('No scene manifest is available.');
      return;
    }
    const version = ++this.loadVersion;
    this.abortController?.abort();
    this.abortController = new AbortController();
    this.setStatus('Loading 3D preview…');
    try {
      const response = await fetch(this.manifestUrl, {
        cache: 'no-store',
        credentials: 'same-origin',
        signal: this.abortController.signal,
        headers: { Accept: 'application/json' },
      });
      if (!response.ok) throw new Error(`Scene request failed (${response.status}).`);
      const manifest = await response.json();
      if (version !== this.loadVersion) return;
      await this.applyManifest(manifest, version);
    } catch (error) {
      if (error?.name === 'AbortError' || version !== this.loadVersion) return;
      this.setStatus(error instanceof Error ? error.message : 'Unable to load the scene.', 'error');
    }
  }

  async applyManifest(manifest, version) {
    if (!manifest || !Number.isInteger(manifest.revision) || manifest.revision < 0) {
      throw new Error('The scene manifest is invalid.');
    }
    this.clearGeometry();
    const loaders = [];
    if (manifest.target_url) {
      loaders.push(this.loadStl('target', manifest.target_url, 0xb8c5d6, 0.9, true, version));
    }
    if (manifest.residual_url) {
      loaders.push(this.loadStl('residual', manifest.residual_url, 0xf87171, 0.45, false, version));
    }
    if (manifest.stock) this.addStock(manifest.stock);
    if (Array.isArray(manifest.supports)) this.addSupports(manifest.supports, manifest.stock);
    if (Array.isArray(manifest.toolpaths)) {
      for (const record of manifest.toolpaths) loaders.push(this.loadToolpaths(record, version));
    }
    await Promise.all(loaders);
    if (version !== this.loadVersion) return;
    this.buildToolFilters();
    this.updateVisibility();
    const hasGeometry = [...this.layerGroups.values()].some((group) => group.children.length > 0);
    if (this.emptyState) this.emptyState.hidden = hasGeometry;
    this.setStatus(
      hasGeometry || this.emptyState ? '' : 'No preview geometry has been generated yet.',
    );
    if (hasGeometry) this.fitCamera();
  }

  async loadStl(layer, url, color, opacity, pickable, version) {
    const loader = new STLLoader();
    const geometry = await loader.loadAsync(url);
    if (version !== this.loadVersion) {
      geometry.dispose();
      return;
    }
    geometry.computeVertexNormals();
    const material = new THREE.MeshStandardMaterial({
      color, opacity, transparent: opacity < 1, side: THREE.DoubleSide,
      roughness: 0.75, metalness: 0.05, depthWrite: opacity >= 0.8,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = layer;
    this.layerGroups.get(layer).add(mesh);
    if (pickable) this.targetObjects.push(mesh);
  }

  addStock(stock) {
    let geometry;
    if (stock.kind === 'cylinder' && finiteNumber(stock.length) && finiteNumber(stock.diameter)) {
      geometry = new THREE.CylinderGeometry(stock.diameter / 2, stock.diameter / 2, stock.length, 64);
      geometry.rotateZ(Math.PI / 2);
    } else if (
      stock.kind === 'rectangle' && finiteNumber(stock.length)
      && finiteNumber(stock.width) && finiteNumber(stock.height)
    ) {
      geometry = new THREE.BoxGeometry(stock.length, stock.width, stock.height);
    } else {
      throw new Error('The analytical stock description is invalid.');
    }
    geometry.translate(stock.length / 2, 0, 0);
    const material = new THREE.MeshStandardMaterial({
      color: 0x3b82f6, transparent: true, opacity: 0.2,
      side: THREE.DoubleSide, depthWrite: false, wireframe: false,
    });
    this.layerGroups.get('stock').add(new THREE.Mesh(geometry, material));
  }

  stockRadius(stock, angle) {
    if (!stock) return 0;
    if (stock.kind === 'cylinder') return stock.diameter / 2;
    const cosine = Math.abs(Math.cos(angle));
    const sine = Math.abs(Math.sin(angle));
    const yLimit = cosine < 1e-12 ? Infinity : stock.width / 2 / cosine;
    const zLimit = sine < 1e-12 ? Infinity : stock.height / 2 / sine;
    return Math.min(yLimit, zLimit);
  }

  addSupports(supports, stock) {
    const group = this.layerGroups.get('supports');
    for (const support of supports) {
      const angleDeg = support.angle_deg;
      if (!finiteNumber(support.x) || !finiteNumber(angleDeg) || !finiteNumber(support.thickness)) continue;
      const angle = THREE.MathUtils.degToRad(angleDeg);
      const radius = this.stockRadius(stock, angle) + support.thickness / 2;
      let geometry;
      if (support.support_type === 'rectangle' || support.type === 'rectangle') {
        if (!finiteNumber(support.length_x) || !finiteNumber(support.width_surface)) continue;
        geometry = new THREE.BoxGeometry(support.length_x, support.thickness, support.width_surface);
      } else if (support.support_type === 'cylinder' || support.type === 'cylinder') {
        if (!finiteNumber(support.diameter)) continue;
        geometry = new THREE.CylinderGeometry(
          support.diameter / 2, support.diameter / 2, support.thickness, 24,
        );
      } else {
        continue;
      }
      geometry.rotateX(angle);
      geometry.translate(support.x, radius * Math.cos(angle), radius * Math.sin(angle));
      const material = new THREE.MeshStandardMaterial({
        color: 0xfb923c, transparent: true, opacity: support.enabled === false ? 0.3 : 0.85,
      });
      group.add(new THREE.Mesh(geometry, material));
    }
  }

  async loadToolpaths(record, version) {
    if (!record || !Number.isInteger(record.tool_number) || !record.buffer_url) {
      throw new Error('A toolpath record is invalid.');
    }
    const response = await fetch(record.buffer_url, {
      cache: 'no-store', credentials: 'same-origin', signal: this.abortController.signal,
    });
    if (!response.ok) throw new Error(`Toolpath request failed (${response.status}).`);
    const buffer = await response.arrayBuffer();
    if (version !== this.loadVersion) return;
    if (buffer.byteLength % 24 !== 0) throw new Error('A toolpath buffer has an invalid byte length.');
    const positions = new Float32Array(buffer);
    for (const value of positions) {
      if (!Number.isFinite(value)) throw new Error('A toolpath buffer contains non-finite values.');
    }
    const expectedSegments = Number(record.segment_count);
    if (!Number.isInteger(expectedSegments) || positions.length !== expectedSegments * 6) {
      throw new Error('A toolpath buffer does not match its manifest.');
    }
    if (expectedSegments === 0) return;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.computeBoundingSphere();
    const color = /^#[0-9a-f]{6}$/i.test(record.color || '') ? record.color : '#56d364';
    const lines = new THREE.LineSegments(
      geometry,
      new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.95 }),
    );
    lines.userData.toolNumber = record.tool_number;
    lines.userData.operationIndex = record.operation_index;
    lines.name = record.operation_name || `Operation ${record.operation_index + 1}`;
    this.layerGroups.get('toolpaths').add(lines);
    this.toolpathObjects.push(lines);
    if (!this.toolVisibility.has(record.tool_number)) this.toolVisibility.set(record.tool_number, true);
  }

  buildToolFilters() {
    if (this.toolSelect) {
      const selected = this.toolSelect.value;
      for (const toolNumber of this.toolVisibility.keys()) {
        this.toolVisibility.set(toolNumber, selected === 'all' || selected === String(toolNumber));
      }
      return;
    }
    if (!this.toolFilterRoot) return;
    this.toolFilterRoot.replaceChildren();
    if (this.toolVisibility.size === 0) return;
    const heading = element('strong', 'scene-tool-filter-heading', 'Tools');
    heading.style.display = 'block';
    heading.style.marginTop = '5px';
    this.toolFilterRoot.append(heading);
    const numbers = [...this.toolVisibility.keys()].sort((left, right) => left - right);
    for (const toolNumber of numbers) {
      this.toolFilterRoot.append(this.checkbox(`Tool ${toolNumber}`, true, (visible) => {
        this.toolVisibility.set(toolNumber, visible);
        this.updateVisibility();
      }));
    }
  }

  updateVisibility() {
    for (const [layer, group] of this.layerGroups) {
      group.visible = this.layerVisibility.get(layer) !== false;
    }
    for (const lines of this.toolpathObjects) {
      lines.visible = this.toolVisibility.get(lines.userData.toolNumber) !== false;
    }
  }

  visibleBounds() {
    const bounds = new THREE.Box3();
    this.scene.updateMatrixWorld(true);
    for (const group of this.layerGroups.values()) {
      group.traverseVisible((object) => {
        if (object.geometry) bounds.union(new THREE.Box3().setFromObject(object));
      });
    }
    return bounds;
  }

  fitCamera(direction = null) {
    const bounds = this.visibleBounds();
    if (bounds.isEmpty()) return;
    const center = bounds.getCenter(new THREE.Vector3());
    const size = bounds.getSize(new THREE.Vector3());
    const radius = Math.max(size.length() / 2, 1);
    let view = direction;
    if (!view) view = this.camera.position.clone().sub(this.controls.target).normalize();
    if (!view.lengthSq()) view = new THREE.Vector3(1, -1, 0.7).normalize();
    const verticalFov = THREE.MathUtils.degToRad(this.camera.fov);
    const distance = radius / Math.sin(verticalFov / 2) * 1.15;
    this.camera.position.copy(center).add(view.clone().normalize().multiplyScalar(distance));
    this.camera.near = Math.max(distance / 1000, 0.001);
    this.camera.far = Math.max(distance * 20, 1000);
    this.camera.updateProjectionMatrix();
    this.controls.target.copy(center);
    this.controls.update();
  }

  orient(axis) {
    const direction = axis === 'X'
      ? new THREE.Vector3(1, 0, 0)
      : axis === 'Y' ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(0, 0, 1);
    this.camera.up.set(0, 0, axis === 'Z' ? 0 : 1);
    if (axis === 'Z') this.camera.up.set(0, 1, 0);
    this.fitCamera(direction);
  }

  async pickTarget(event) {
    if (this.targetObjects.length === 0) return;
    const rect = this.renderer.domElement.getBoundingClientRect();
    const pointer = new THREE.Vector2(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1,
    );
    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(pointer, this.camera);
    const intersection = raycaster.intersectObjects(this.targetObjects, false)[0];
    if (!intersection) return;
    const { x, y, z } = intersection.point;
    if (![x, y, z].every(Number.isFinite)) return;
    const angleDeg = (THREE.MathUtils.radToDeg(Math.atan2(z, y)) + 360) % 360;
    const detail = { x, y, z, angleDeg };
    this.container.dispatchEvent(new CustomEvent('rotarycam:support-picked', {
      bubbles: true, detail,
    }));
    if (!this.supportPickUrl) return;
    const csrfToken = this.container.dataset.csrfToken
      || document.querySelector('meta[name="csrf-token"]')?.content
      || document.querySelector('input[name="csrf_token"]')?.value
      || '';
    try {
      const response = await fetch(this.supportPickUrl, {
        method: 'POST', credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': csrfToken },
        body: JSON.stringify({ x, y, z, csrf_token: csrfToken }),
      });
      if (!response.ok) throw new Error(`Support pick failed (${response.status}).`);
      if (window.htmx?.ajax) {
        await window.htmx.ajax('GET', `${window.location.pathname}?step=supports`, {
          target: '#workspace-shell', swap: 'outerHTML',
        });
        document.dispatchEvent(new CustomEvent('rotarycam:scene-changed', {
          detail: { projectId: this.container.dataset.projectId },
        }));
      } else {
        window.location.reload();
      }
    } catch (error) {
      this.setStatus(error instanceof Error ? error.message : 'Unable to save the support.', 'error');
    }
  }
}

let viewer = null;

function initializeViewer() {
  const container = document.querySelector('#scene-viewer');
  if (!container) return;
  if (!viewer || viewer.container !== container) viewer = new RotaryCamViewer(container);
  viewer.load();
}

export { RotaryCamViewer, initializeViewer };
