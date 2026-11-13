import React, { useEffect, useRef, useState, useMemo } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import {
  RotateCcw,
  Maximize2,
  Minimize2,
  Camera,
  Play,
  Pause,
  Layers,
  Eye,
  EyeOff,
  Sliders,
  Scissors,
  Box,
  Compass,
  Sparkles,
  Info
} from 'lucide-react';

export default function ThreeDViewer({ data, modality = 'FLAIR' }) {
  const mountRef = useRef(null);
  const sceneRef = useRef(null);
  const rendererRef = useRef(null);
  const cameraRef = useRef(null);
  const controlsRef = useRef(null);

  // Mesh refs for dynamic property updates
  const brainMeshRef = useRef(null);
  const wtMeshRef = useRef(null);
  const tcMeshRef = useRef(null);
  const etMeshRef = useRef(null);
  const gridHelperRef = useRef(null);
  const boundingBoxRef = useRef(null);

  // Slicing planes refs
  const axialPlaneMeshRef = useRef(null);
  const coronalPlaneMeshRef = useRef(null);
  const sagittalPlaneMeshRef = useRef(null);
  const clipPlanesRef = useRef({
    axial: new THREE.Plane(new THREE.Vector3(0, -1, 0), 35),
    coronal: new THREE.Plane(new THREE.Vector3(0, 0, -1), 35),
    sagittal: new THREE.Plane(new THREE.Vector3(-1, 0, 0), 35)
  });

  // UI State
  const [renderMode, setRenderMode] = useState('hologram'); // hologram | solid | wireframe
  const [isRotating, setIsRotating] = useState(false);
  const isRotatingRef = useRef(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showBrain, setShowBrain] = useState(true);
  const [showWT, setShowWT] = useState(true);
  const [showTC, setShowTC] = useState(true);
  const [showET, setShowET] = useState(true);
  const [showGrid, setShowGrid] = useState(true);
  const [showPlanes, setShowPlanes] = useState(true);
  const [enableClipping, setEnableClipping] = useState(false);

  // Sliders
  const [brainOpacity, setBrainOpacity] = useState(0.35);
  const [tumorOpacity, setTumorOpacity] = useState(0.85);
  const [sliceZ, setSliceZ] = useState(0); // -30 to +30
  const [sliceY, setSliceY] = useState(0); // -30 to +30
  const [sliceX, setSliceX] = useState(0); // -30 to +30

  // Hover Coordinates
  const [hoverCoords, setHoverCoords] = useState(null);

  // Calculated volumes
  const volumes = useMemo(() => {
    if (!data) {
      return { wt: 0, tc: 0, et: 0, total: 0 };
    }
    return {
      wt: data.volumes_cm3?.['Whole Tumor (WT)'] ?? 0,
      tc: data.volumes_cm3?.['Tumor Core (TC)'] ?? 0,
      et: data.volumes_cm3?.['Enhancing Tumor (ET)'] ?? 0,
      total: data.total_tumor_volume_cm3 ?? 0
    };
  }, [data]);

  // Generate Procedural Realistic Brain Geometry with Gyri/Sulci
  const createBrainGeometry = () => {
    const geom = new THREE.SphereGeometry(32, 64, 48);
    const pos = geom.attributes.position;
    const v = new THREE.Vector3();

    for (let i = 0; i < pos.count; i++) {
      v.fromBufferAttribute(pos, i);

      // Elongate into anatomical cranial shape: AP (Z) > Lat (X) > Vert (Y)
      v.z *= 1.25; // Antero-posterior elongation
      v.y *= 1.05; // Superior-inferior height
      v.x *= 0.98; // Bi-temporal width

      // Longitudinal cerebral fissure indentation (groove between left & right hemispheres)
      const distFromMidline = Math.abs(v.x);
      if (distFromMidline < 12 && v.y > 0) {
        const fissureFactor = 1.0 - Math.exp(-Math.pow(distFromMidline / 6, 2));
        v.x = Math.sign(v.x) * (distFromMidline * 0.85 + 0.5);
        v.y -= (1.0 - fissureFactor) * 3.5;
      }

      // Temporal lobe anterior inferior projection
      if (v.y < -5 && v.z > -10 && v.z < 20) {
        v.x *= 1.08;
      }

      // Procedural gyri & sulci high-frequency anatomical wrinkles
      const freq1 = 0.25;
      const freq2 = 0.5;
      const noise1 = Math.sin(v.x * freq1) * Math.cos(v.y * freq1) * Math.sin(v.z * freq1);
      const noise2 = Math.sin(v.x * freq2 + 1.2) * Math.cos(v.z * freq2 + 0.8) * 0.5;
      const wrinkle = (noise1 + noise2) * 1.4;

      v.addScaledVector(v.clone().normalize(), wrinkle);

      pos.setXYZ(i, v.x, v.y, v.z);
    }

    geom.computeVertexNormals();
    return geom;
  };

  // Generate Organic Tumor Mesh with Multi-Octave Deformations
  const createTumorGeometry = (baseRadius, deformationFactor, seedOffset = 0) => {
    const geom = new THREE.IcosahedronGeometry(baseRadius, 5);
    const pos = geom.attributes.position;
    const v = new THREE.Vector3();

    for (let i = 0; i < pos.count; i++) {
      v.fromBufferAttribute(pos, i);

      // Organic multi-directional tumor lobulations
      const n1 = Math.sin(v.x * 0.2 + seedOffset) * Math.cos(v.y * 0.2) * Math.sin(v.z * 0.2);
      const n2 = Math.cos(v.x * 0.45) * Math.sin(v.z * 0.45 + seedOffset) * 0.5;
      const displacement = (n1 + n2) * deformationFactor;

      v.addScaledVector(v.clone().normalize(), displacement);

      // Slight anisotropic elongation along white matter tracts
      v.x *= 1.05;
      v.y *= 0.92;
      v.z *= 1.12;

      pos.setXYZ(i, v.x, v.y, v.z);
    }

    geom.computeVertexNormals();
    return geom;
  };

  // Convert raw Marching Cubes vertex/face buffer into Three.js BufferGeometry
  const buildBufferGeometryFromMesh = (meshData) => {
    if (!meshData || !meshData.vertices || meshData.vertices.length === 0) return null;
    try {
      const geom = new THREE.BufferGeometry();
      geom.setAttribute('position', new THREE.Float32BufferAttribute(meshData.vertices, 3));
      if (meshData.faces && meshData.faces.length > 0) {
        geom.setIndex(meshData.faces);
      }
      // Always recompute normals — MC raw normals can have winding issues in Three.js
      geom.computeVertexNormals();
      return geom;
    } catch (e) {
      console.warn('Error constructing buffer geometry from Marching Cubes mesh:', e);
      return null;
    }
  };

  // Create Orthogonal Slice Texture
  const createSliceTexture = (type) => {
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 256;
    const ctx = canvas.getContext('2d');

    // Background
    ctx.fillStyle = '#050811';
    ctx.fillRect(0, 0, 256, 256);

    // Anatomical contour
    ctx.beginPath();
    ctx.ellipse(128, 128, 98, 115, 0, 0, Math.PI * 2);
    ctx.fillStyle = '#141e33';
    ctx.fill();
    ctx.strokeStyle = '#253556';
    ctx.lineWidth = 3;
    ctx.stroke();

    // Internal tissue lines
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.15)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(128, 20);
    ctx.lineTo(128, 236);
    ctx.stroke();

    // Ventricles
    ctx.fillStyle = '#060913';
    ctx.beginPath();
    ctx.ellipse(112, 125, 10, 26, -0.15, 0, Math.PI * 2);
    ctx.ellipse(144, 125, 10, 26, 0.15, 0, Math.PI * 2);
    ctx.fill();

    // Tumor slice intersect
    ctx.beginPath();
    ctx.ellipse(156, 105, 24, 20, 0.3, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(56, 189, 248, 0.35)';
    ctx.fill();
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.beginPath();
    ctx.ellipse(156, 105, 14, 12, 0.3, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(251, 191, 36, 0.55)';
    ctx.fill();

    ctx.beginPath();
    ctx.ellipse(158, 104, 8, 7, 0.3, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(244, 63, 94, 0.7)';
    ctx.fill();

    const texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter;
    return texture;
  };

  // Initialize Three.js Scene
  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    const width = container.clientWidth;
    const height = container.clientHeight || 480;

    // 1. Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#070b14');
    sceneRef.current = scene;

    // 2. Camera
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.set(70, 48, 85);
    cameraRef.current = camera;

    // 3. Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.localClippingEnabled = true;
    container.innerHTML = '';
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // 4. Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxDistance = 250;
    controls.minDistance = 25;
    controls.target.set(0, 0, 0);
    controls.saveState();
    controlsRef.current = controls;

    // 5. Lighting Setup (Medical Surgical Studio)
    const ambientLight = new THREE.AmbientLight(0x0f172a, 1.8);
    scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0xffffff, 2.0);
    dirLight1.position.set(60, 80, 50);
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0x38bdf8, 1.4);
    dirLight2.position.set(-60, -30, -50);
    scene.add(dirLight2);

    const rimLight = new THREE.DirectionalLight(0x818cf8, 1.2);
    rimLight.position.set(0, 50, -80);
    scene.add(rimLight);

    const centerPointLight = new THREE.PointLight(0x38bdf8, 1.2, 120);
    centerPointLight.position.set(12, 6, 8);
    scene.add(centerPointLight);

    // 6. Stereotactic Ground Grid & Coordinate Cage
    const gridHelper = new THREE.GridHelper(90, 18, 0x1e3a5f, 0x0f2038);
    gridHelper.position.y = -38;
    scene.add(gridHelper);
    gridHelperRef.current = gridHelper;

    // Bounding Box Cage (64x64x64 mm Isotropic Volume)
    const boxGeom = new THREE.BoxGeometry(68, 68, 68);
    const boxEdges = new THREE.EdgesGeometry(boxGeom);
    const boxLine = new THREE.LineSegments(
      boxEdges,
      new THREE.LineBasicMaterial({ color: 0x22416b, transparent: true, opacity: 0.4 })
    );
    scene.add(boxLine);
    boundingBoxRef.current = boxLine;

    // 7. Brain Cortex Mesh
    const brainGeom = createBrainGeometry();
    const brainMat = new THREE.MeshPhysicalMaterial({
      color: 0x38bdf8,
      emissive: 0x0c2545,
      roughness: 0.25,
      metalness: 0.1,
      transmission: 0.65,
      thickness: 1.2,
      transparent: true,
      opacity: brainOpacity,
      side: THREE.DoubleSide,
      depthWrite: false,
      clippingPlanes: []
    });
    const brainMesh = new THREE.Mesh(brainGeom, brainMat);
    brainMesh.renderOrder = 4;
    scene.add(brainMesh);
    brainMeshRef.current = brainMesh;

    // 8. Tumor Sub-Regions (BraTS: WT, TC, ET)
    const tumorCenter = new THREE.Vector3(12, 6, 8); // Right fronto-parietal focus

    // Scaling based on volume in cm³
    const wtRad = Math.max(9, Math.min(18, Math.cbrt(volumes.wt) * 4.6));
    const tcRad = Math.max(6, Math.min(wtRad * 0.78, Math.cbrt(volumes.tc) * 4.0));
    const etRad = Math.max(3.5, Math.min(tcRad * 0.68, Math.cbrt(volumes.et) * 3.4));

    // A. Whole Tumor (WT) - Translucent Edema Shell (RenderOrder 3 so it blends over inner cores)
    const wtGeom = createTumorGeometry(wtRad, 3.2, 1.5);
    const wtMat = new THREE.MeshPhysicalMaterial({
      color: 0x38bdf8,
      emissive: 0x0284c7,
      emissiveIntensity: 0.35,
      roughness: 0.4,
      transparent: true,
      opacity: tumorOpacity * 0.45,
      depthWrite: false, // Translucent envelope does not block inner core
      side: THREE.DoubleSide,
      clippingPlanes: []
    });
    const wtMesh = new THREE.Mesh(wtGeom, wtMat);
    wtMesh.renderOrder = 3;
    wtMesh.position.copy(tumorCenter);
    wtMesh.visible = !!data && volumes.wt > 0;
    scene.add(wtMesh);
    wtMeshRef.current = wtMesh;

    // B. Tumor Core (TC) - Solid Amber Gold Core
    const tcGeom = createTumorGeometry(tcRad, 2.2, 4.2);
    const tcMat = new THREE.MeshPhysicalMaterial({
      color: 0xfbbf24,
      emissive: 0xd97706,
      emissiveIntensity: 0.55,
      roughness: 0.25,
      transparent: true,
      opacity: Math.max(0.85, tumorOpacity),
      depthWrite: true,
      polygonOffset: true,
      polygonOffsetFactor: -1,
      polygonOffsetUnits: -1,
      side: THREE.DoubleSide,
      clippingPlanes: []
    });
    const tcMesh = new THREE.Mesh(tcGeom, tcMat);
    tcMesh.renderOrder = 2;
    tcMesh.position.copy(tumorCenter);
    tcMesh.visible = !!data && volumes.tc > 0;
    scene.add(tcMesh);
    tcMeshRef.current = tcMesh;

    // C. Enhancing Tumor (ET) - Solid Crimson Rose Active Core (RenderOrder 1)
    const etGeom = createTumorGeometry(etRad, 1.6, 7.8);
    const etMat = new THREE.MeshPhysicalMaterial({
      color: 0xf43f5e,
      emissive: 0xe11d48,
      emissiveIntensity: 0.85,
      roughness: 0.2,
      transparent: true,
      opacity: Math.max(0.95, tumorOpacity),
      depthWrite: true,
      polygonOffset: true,
      polygonOffsetFactor: -2,
      polygonOffsetUnits: -2,
      side: THREE.DoubleSide,
      clippingPlanes: []
    });
    const etMesh = new THREE.Mesh(etGeom, etMat);
    etMesh.renderOrder = 1;
    etMesh.position.copy(tumorCenter);
    etMesh.visible = !!data && volumes.et > 0;
    scene.add(etMesh);
    etMeshRef.current = etMesh;

    // 9. Orthogonal Slicing MPR Planes
    const planeGeom = new THREE.PlaneGeometry(64, 64);

    // Axial Plane (XY Plane movable along Z)
    const axialMat = new THREE.MeshBasicMaterial({
      map: createSliceTexture('axial'),
      transparent: true,
      opacity: 0.55,
      side: THREE.DoubleSide,
      depthWrite: false
    });
    const axialMesh = new THREE.Mesh(planeGeom, axialMat);
    axialMesh.rotation.x = Math.PI / 2;
    scene.add(axialMesh);
    axialPlaneMeshRef.current = axialMesh;

    // Coronal Plane (XZ Plane movable along Y)
    const coronalMat = new THREE.MeshBasicMaterial({
      map: createSliceTexture('coronal'),
      transparent: true,
      opacity: 0.45,
      side: THREE.DoubleSide,
      depthWrite: false
    });
    const coronalMesh = new THREE.Mesh(planeGeom, coronalMat);
    scene.add(coronalMesh);
    coronalPlaneMeshRef.current = coronalMesh;

    // Sagittal Plane (YZ Plane movable along X)
    const sagittalMat = new THREE.MeshBasicMaterial({
      map: createSliceTexture('sagittal'),
      transparent: true,
      opacity: 0.45,
      side: THREE.DoubleSide,
      depthWrite: false
    });
    const sagittalMesh = new THREE.Mesh(planeGeom, sagittalMat);
    sagittalMesh.rotation.y = Math.PI / 2;
    scene.add(sagittalMesh);
    sagittalPlaneMeshRef.current = sagittalMesh;

    // 10. Animation Loop
    let animId;
    const clock = new THREE.Clock();

    const animate = () => {
      animId = requestAnimationFrame(animate);
      const delta = clock.getDelta();
      const time = clock.getElapsedTime();

      // Auto rotation
      if (controlsRef.current) {
        controlsRef.current.autoRotate = isRotatingRef.current;
        controlsRef.current.autoRotateSpeed = 2.0;
      }

      // Gentle pulsing of Enhancing Tumor (angiogenic signature)
      if (etMeshRef.current) {
        const pulse = 1.0 + Math.sin(time * 3.5) * 0.035;
        etMeshRef.current.scale.set(pulse, pulse, pulse);
      }

      controls.update();
      renderer.render(scene, camera);
    };

    animate();

    // Resize Handler
    const handleResize = () => {
      if (!container) return;
      const w = container.clientWidth;
      const h = container.clientHeight || 480;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };

    window.addEventListener('resize', handleResize);

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener('resize', handleResize);
      renderer.dispose();
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
    };
  }, []);

  // Update Volumes & Geometries when `data` changes
  useEffect(() => {
    if (!wtMeshRef.current || !tcMeshRef.current || !etMeshRef.current) return;

    // Only use real Marching Cubes meshes when prediction quality is good
    const predQuality = data?.prediction_quality ?? 'low';
    const hasRealMeshes = predQuality === 'good' && data?.meshes && (
      (data.meshes.wt?.vertex_count > 0) ||
      (data.meshes.tc?.vertex_count > 0) ||
      (data.meshes.et?.vertex_count > 0)
    );

    if (hasRealMeshes) {
      // 1. Whole Tumor (WT)
      if (data.meshes.wt?.vertex_count > 0) {
        const wtGeom = buildBufferGeometryFromMesh(data.meshes.wt);
        if (wtGeom) {
          wtMeshRef.current.geometry.dispose();
          wtMeshRef.current.geometry = wtGeom;
          wtMeshRef.current.position.set(0, 0, 0);
          wtMeshRef.current.renderOrder = 3;
          wtMeshRef.current.material.depthWrite = false;
          wtMeshRef.current.material.transparent = true;
          wtMeshRef.current.material.opacity = tumorOpacity * 0.45;
          wtMeshRef.current.visible = showWT;
        }
      } else {
        wtMeshRef.current.visible = false;
      }

      // 2. Tumor Core (TC) - Solid Amber Core with polygonOffset to stop Z-fighting
      if (data.meshes.tc?.vertex_count > 0) {
        const tcGeom = buildBufferGeometryFromMesh(data.meshes.tc);
        if (tcGeom) {
          tcMeshRef.current.geometry.dispose();
          tcMeshRef.current.geometry = tcGeom;
          tcMeshRef.current.position.set(0, 0, 0);
          tcMeshRef.current.renderOrder = 2;
          tcMeshRef.current.material.depthWrite = true;
          tcMeshRef.current.material.polygonOffset = true;
          tcMeshRef.current.material.polygonOffsetFactor = -1;
          tcMeshRef.current.material.polygonOffsetUnits = -1;
          tcMeshRef.current.material.opacity = Math.max(0.85, tumorOpacity);
          tcMeshRef.current.visible = showTC;
        }
      } else {
        tcMeshRef.current.visible = false;
      }

      // 3. Enhancing Tumor (ET) - Solid Crimson Rim with polygonOffset
      if (data.meshes.et?.vertex_count > 0) {
        const etGeom = buildBufferGeometryFromMesh(data.meshes.et);
        if (etGeom) {
          etMeshRef.current.geometry.dispose();
          etMeshRef.current.geometry = etGeom;
          etMeshRef.current.position.set(0, 0, 0);
          etMeshRef.current.renderOrder = 1;
          etMeshRef.current.material.depthWrite = true;
          etMeshRef.current.material.polygonOffset = true;
          etMeshRef.current.material.polygonOffsetFactor = -2;
          etMeshRef.current.material.polygonOffsetUnits = -2;
          etMeshRef.current.material.opacity = Math.max(0.95, tumorOpacity);
          etMeshRef.current.visible = showET;
        }
      } else {
        etMeshRef.current.visible = false;
      }

      // Orient camera towards tumor centroid
      if (data?.centroid && controlsRef.current) {
        controlsRef.current.target.set(
          data.centroid[0] * 0.3,
          data.centroid[1] * 0.3,
          data.centroid[2] * 0.3
        );
        controlsRef.current.update();
      }
    } else if (data && volumes.wt <= 0) {
      // Explicitly zero tumor predicted — hide tumor subregions
      wtMeshRef.current.visible = false;
      tcMeshRef.current.visible = false;
      etMeshRef.current.visible = false;
    } else if (!data) {
      // Clear state (e.g. user clicked Clear) — hide all tumor subregions from 3D skull!
      wtMeshRef.current.visible = false;
      tcMeshRef.current.visible = false;
      etMeshRef.current.visible = false;
      if (controlsRef.current) {
        controlsRef.current.target.set(0, 0, 0);
        controlsRef.current.update();
      }
    } else {
      // Procedural fallback — scaled by BraTS-realistic volume ranges
      // Clamp volumes to clinically plausible ranges so the mesh fits inside the brain
      const wtVol = Math.min(volumes.wt || 1, 80);
      const tcVol = Math.min(volumes.tc || 1, Math.min(wtVol, 40));
      const etVol = Math.min(volumes.et || 1, Math.min(tcVol, 20));

      const wtRad = Math.max(8, Math.min(20, Math.cbrt(wtVol) * 4.2));
      const tcRad = Math.max(5, Math.min(wtRad * 0.78, Math.cbrt(tcVol) * 3.8));
      const etRad = Math.max(3, Math.min(tcRad * 0.68, Math.cbrt(etVol) * 3.2));

      // Use real centroid if provided, otherwise default focal point
      const centerPos = (data?.centroid && (data.centroid[0] !== 0 || data.centroid[1] !== 0 || data.centroid[2] !== 0))
        ? new THREE.Vector3(data.centroid[0], data.centroid[1], data.centroid[2])
        : new THREE.Vector3(10, 4, 6);

      wtMeshRef.current.geometry.dispose();
      wtMeshRef.current.geometry = createTumorGeometry(wtRad, 3.2, 1.5);
      wtMeshRef.current.position.copy(centerPos);
      wtMeshRef.current.visible = showWT;

      tcMeshRef.current.geometry.dispose();
      tcMeshRef.current.geometry = createTumorGeometry(tcRad, 2.2, 4.2);
      tcMeshRef.current.position.copy(centerPos);
      tcMeshRef.current.visible = showTC;

      etMeshRef.current.geometry.dispose();
      etMeshRef.current.geometry = createTumorGeometry(etRad, 1.6, 7.8);
      etMeshRef.current.position.copy(centerPos);
      etMeshRef.current.visible = showET;
    }
  }, [data, volumes]);

  // Update Render Modes (Hologram / Solid / Wireframe)
  useEffect(() => {
    if (!brainMeshRef.current) return;
    const mat = brainMeshRef.current.material;

    if (renderMode === 'wireframe') {
      mat.wireframe = true;
      mat.opacity = 0.25;
      mat.color.set(0x38bdf8);
    } else if (renderMode === 'solid') {
      mat.wireframe = false;
      mat.opacity = 0.85;
      mat.transmission = 0.05;
      mat.color.set(0x233758);
    } else {
      // Hologram / Glass
      mat.wireframe = false;
      mat.opacity = brainOpacity;
      mat.transmission = 0.7;
      mat.color.set(0x38bdf8);
    }
  }, [renderMode, brainOpacity]);

  // Update Opacity & Visibility
  useEffect(() => {
    if (brainMeshRef.current) {
      brainMeshRef.current.visible = showBrain;
      if (renderMode !== 'solid' && renderMode !== 'wireframe') {
        brainMeshRef.current.material.opacity = brainOpacity;
      }
    }
    if (wtMeshRef.current) {
      wtMeshRef.current.visible = showWT;
      wtMeshRef.current.material.opacity = tumorOpacity * 0.75;
    }
    if (tcMeshRef.current) {
      tcMeshRef.current.visible = showTC;
      tcMeshRef.current.material.opacity = tumorOpacity * 0.9;
    }
    if (etMeshRef.current) {
      etMeshRef.current.visible = showET;
      etMeshRef.current.material.opacity = tumorOpacity;
    }
    if (gridHelperRef.current) gridHelperRef.current.visible = showGrid;
    if (boundingBoxRef.current) boundingBoxRef.current.visible = showGrid;

    if (axialPlaneMeshRef.current) axialPlaneMeshRef.current.visible = showPlanes;
    if (coronalPlaneMeshRef.current) coronalPlaneMeshRef.current.visible = showPlanes;
    if (sagittalPlaneMeshRef.current) sagittalPlaneMeshRef.current.visible = showPlanes;
  }, [showBrain, showWT, showTC, showET, showGrid, showPlanes, brainOpacity, tumorOpacity]);

  // Update MPR Plane Positions
  useEffect(() => {
    if (axialPlaneMeshRef.current) axialPlaneMeshRef.current.position.y = sliceZ;
    if (coronalPlaneMeshRef.current) coronalPlaneMeshRef.current.position.z = sliceY;
    if (sagittalPlaneMeshRef.current) sagittalPlaneMeshRef.current.position.x = sliceX;

    // Update clipping planes if active
    if (enableClipping && brainMeshRef.current && wtMeshRef.current) {
      const planes = [
        new THREE.Plane(new THREE.Vector3(0, -1, 0), sliceZ + 0.1),
        new THREE.Plane(new THREE.Vector3(0, 0, -1), sliceY + 0.1)
      ];
      brainMeshRef.current.material.clippingPlanes = planes;
      wtMeshRef.current.material.clippingPlanes = planes;
      tcMeshRef.current.material.clippingPlanes = planes;
      etMeshRef.current.material.clippingPlanes = planes;
    } else if (brainMeshRef.current && wtMeshRef.current) {
      brainMeshRef.current.material.clippingPlanes = [];
      wtMeshRef.current.material.clippingPlanes = [];
      tcMeshRef.current.material.clippingPlanes = [];
      etMeshRef.current.material.clippingPlanes = [];
    }
  }, [sliceZ, sliceY, sliceX, enableClipping]);

  // Sync Auto-Rotation ref and controls dynamically
  useEffect(() => {
    isRotatingRef.current = isRotating;
    if (controlsRef.current) {
      controlsRef.current.autoRotate = isRotating;
      controlsRef.current.autoRotateSpeed = 2.0;
    }
  }, [isRotating]);

  // Set Camera Orientation Presets
  const setCameraPreset = (view) => {
    if (!cameraRef.current || !controlsRef.current) return;
    const cam = cameraRef.current;
    const dist = 110;

    // Stop auto-rotation when switching to a fixed projection
    setIsRotating(false);
    isRotatingRef.current = false;
    if (controlsRef.current) controlsRef.current.autoRotate = false;

    if (view === 'axial') {
      // Superior (Top-down)
      cam.position.set(0, dist, 0.001);
      cam.up.set(0, 0, -1);
    } else if (view === 'coronal') {
      // Anterior (Frontal)
      cam.position.set(0, 0, dist);
      cam.up.set(0, 1, 0);
    } else if (view === 'sagittal') {
      // Lateral (Side)
      cam.position.set(dist, 0, 0);
      cam.up.set(0, 1, 0);
    } else {
      // 3D Isometric Perspective
      cam.position.set(70, 48, 85);
      cam.up.set(0, 1, 0);
    }
    cam.zoom = 1;
    cam.lookAt(0, 0, 0);
    cam.updateProjectionMatrix();
    controlsRef.current.target.set(0, 0, 0);
    controlsRef.current.update();
  };

  // Dedicated Reset Camera View Handler
  const handleResetCamera = () => {
    setIsRotating(false);
    isRotatingRef.current = false;
    if (controlsRef.current && cameraRef.current) {
      controlsRef.current.autoRotate = false;
      controlsRef.current.reset();
      cameraRef.current.position.set(70, 48, 85);
      cameraRef.current.up.set(0, 1, 0);
      cameraRef.current.zoom = 1;
      cameraRef.current.lookAt(0, 0, 0);
      cameraRef.current.updateProjectionMatrix();
      controlsRef.current.target.set(0, 0, 0);
      controlsRef.current.update();
    }
  };

  // High-Resolution Snapshot Capture
  const handleSnapshot = () => {
    if (!rendererRef.current) return;
    const link = document.createElement('a');
    link.download = `NeuroScan_3D_Capture_${data?.patient_id || 'Case'}.png`;
    link.href = rendererRef.current.domElement.toDataURL('image/png');
    link.click();
  };

  // Mouse Move for 3D Stereotactic Coordinates
  const handleMouseMove = (e) => {
    if (!mountRef.current || !cameraRef.current || !sceneRef.current) return;
    const rect = mountRef.current.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    const y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(new THREE.Vector2(x, y), cameraRef.current);

    const intersects = raycaster.intersectObjects([
      brainMeshRef.current,
      wtMeshRef.current,
      tcMeshRef.current
    ].filter(Boolean));

    if (intersects.length > 0) {
      const p = intersects[0].point;
      setHoverCoords({
        x: p.x.toFixed(1),
        y: p.y.toFixed(1),
        z: p.z.toFixed(1),
        obj: intersects[0].object === wtMeshRef.current ? 'Whole Tumor' :
             intersects[0].object === tcMeshRef.current ? 'Tumor Core' : 'Cerebral Cortex'
      });
    } else {
      setHoverCoords(null);
    }
  };

  return (
    <div className={`threed-viewer-container ${isFullscreen ? 'fullscreen' : ''}`}>
      {/* 3D Viewer Header Toolbar */}
      <div className="threed-toolbar-top">
        <div className="threed-brand-tag">
          <Box size={18} color="#38bdf8" />
          <div>
            <span className="threed-title">Interactive 3D Volumetric Studio</span>
            <span className="threed-subtitle">Three.js WebGL / VTK Spatial Pipeline</span>
          </div>
        </div>

        {/* Camera Orientation Presets */}
        <div className="threed-presets-row">
          <button className="preset-btn" onClick={() => setCameraPreset('isometric')}>
            3D Iso
          </button>
          <button className="preset-btn" onClick={() => setCameraPreset('axial')}>
            Axial (Z)
          </button>
          <button className="preset-btn" onClick={() => setCameraPreset('coronal')}>
            Coronal (Y)
          </button>
          <button className="preset-btn" onClick={() => setCameraPreset('sagittal')}>
            Sagittal (X)
          </button>

          <button
            className={`preset-btn icon-btn ${isRotating ? 'active' : ''}`}
            onClick={() => setIsRotating(prev => !prev)}
            title={isRotating ? "Pause Auto Rotation" : "Auto Orbit Rotation"}
          >
            {isRotating ? <Pause size={14} /> : <Play size={14} />}
          </button>

          <button
            className="preset-btn icon-btn"
            onClick={handleResetCamera}
            title="Reset Camera View"
          >
            <RotateCcw size={14} />
          </button>

          <button
            className="preset-btn icon-btn"
            onClick={handleSnapshot}
            title="Export 3D Snapshot PNG"
          >
            <Camera size={14} />
          </button>

          <button
            className="preset-btn icon-btn"
            onClick={() => setIsFullscreen(!isFullscreen)}
            title="Toggle Expanded View"
          >
            {isFullscreen ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
          </button>
        </div>
      </div>

      {/* Main 3D Canvas Stage */}
      <div
        className="threed-canvas-wrapper"
        ref={mountRef}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setHoverCoords(null)}
      >
        {/* Real-time Stereotactic Coordinates HUD */}
        <div className="threed-hud-coords">
          <div className="hud-header">
            <Compass size={12} color="#38bdf8" />
            <span>STEREOTACTIC HUD</span>
          </div>
          {hoverCoords ? (
            <div className="hud-data">
              <div>Target: <strong>{hoverCoords.obj}</strong></div>
              <div>X: <strong>{hoverCoords.x} mm</strong> | Y: <strong>{hoverCoords.y} mm</strong> | Z: <strong>{hoverCoords.z} mm</strong></div>
            </div>
          ) : (
            <div className="hud-data muted">
              Hover over 3D model for coordinates
            </div>
          )}
        </div>

        {/* Anatomical Compass Widget */}
        <div className="threed-hud-axes">
          <div className="axis-item" style={{ color: '#38bdf8' }}>+X: LATERAL</div>
          <div className="axis-item" style={{ color: '#10b981' }}>+Y: SUPERIOR</div>
          <div className="axis-item" style={{ color: '#f59e0b' }}>+Z: ANTERIOR</div>
        </div>

        {/* Volume Dimensions Stamp */}
        <div className="threed-hud-vol-stamp">
          <div>GEOMETRY: <strong>64×64×64 mm³</strong></div>
          <div>ISOTROPIC SPACING: <strong>1.0 mm</strong></div>
        </div>
      </div>

      {/* Controls Drawer & Layer Toggles */}
      <div className="threed-controls-bottom">
        {/* Layer Visibility Toggles */}
        <div className="threed-layers-bar">
          <span className="threed-section-label">
            <Layers size={14} /> Layers:
          </span>

          <button
            className={`layer-chip brain ${showBrain ? 'active' : ''}`}
            onClick={() => setShowBrain(!showBrain)}
          >
            <span className="chip-color" style={{ background: '#38bdf8' }} />
            Brain Cortex {showBrain ? <Eye size={12} /> : <EyeOff size={12} />}
          </button>

          {data && volumes.wt > 0 && (
            <>
              <button
                className={`layer-chip wt ${showWT ? 'active' : ''}`}
                onClick={() => setShowWT(!showWT)}
              >
                <span className="chip-color" style={{ background: '#38bdf8' }} />
                Whole Tumor ({volumes.wt.toFixed(1)} cm³) {showWT ? <Eye size={12} /> : <EyeOff size={12} />}
              </button>

              <button
                className={`layer-chip tc ${showTC ? 'active' : ''}`}
                onClick={() => setShowTC(!showTC)}
              >
                <span className="chip-color" style={{ background: '#fbbf24' }} />
                Tumor Core ({volumes.tc.toFixed(1)} cm³) {showTC ? <Eye size={12} /> : <EyeOff size={12} />}
              </button>

              <button
                className={`layer-chip et ${showET ? 'active' : ''}`}
                onClick={() => setShowET(!showET)}
              >
                <span className="chip-color" style={{ background: '#f43f5e' }} />
                Enhancing Rim ({volumes.et.toFixed(1)} cm³) {showET ? <Eye size={12} /> : <EyeOff size={12} />}
              </button>
            </>
          )}

          <button
            className={`layer-chip ${showPlanes ? 'active' : ''}`}
            onClick={() => setShowPlanes(!showPlanes)}
          >
            MPR Ortho Slices
          </button>

          <button
            className={`layer-chip ${showGrid ? 'active' : ''}`}
            onClick={() => setShowGrid(!showGrid)}
          >
            3D Cage Grid
          </button>

          <button
            className={`layer-chip clip-btn ${enableClipping ? 'active' : ''}`}
            onClick={() => setEnableClipping(!enableClipping)}
          >
            <Scissors size={12} />
            3D Cross-Section Clip
          </button>
        </div>

        {/* Shading Mode & Sliders Bar */}
        <div className="threed-sliders-grid">
          {/* Shading Style */}
          <div className="slider-item">
            <label className="slider-label">
              <span>Shading Shader</span>
              <strong>{renderMode.toUpperCase()}</strong>
            </label>
            <div className="render-mode-tabs">
              {['hologram', 'solid', 'wireframe'].map((mode) => (
                <button
                  key={mode}
                  className={`mode-btn ${renderMode === mode ? 'active' : ''}`}
                  onClick={() => setRenderMode(mode)}
                >
                  {mode}
                </button>
              ))}
            </div>
          </div>

          {/* Brain Opacity Slider */}
          <div className="slider-item">
            <label className="slider-label">
              <span>Cortex Transparency</span>
              <strong>{Math.round(brainOpacity * 100)}%</strong>
            </label>
            <input
              type="range"
              min="0.05"
              max="0.9"
              step="0.05"
              value={brainOpacity}
              onChange={(e) => setBrainOpacity(parseFloat(e.target.value))}
              className="custom-range-slider"
            />
          </div>

          {/* Axial Z-Plane Slice */}
          <div className="slider-item">
            <label className="slider-label">
              <span>Axial Cut (Z)</span>
              <strong>{sliceZ > 0 ? `+${sliceZ}` : sliceZ} mm</strong>
            </label>
            <input
              type="range"
              min="-28"
              max="28"
              step="1"
              value={sliceZ}
              onChange={(e) => setSliceZ(parseInt(e.target.value))}
              className="custom-range-slider"
            />
          </div>

          {/* Coronal Y-Plane Slice */}
          <div className="slider-item">
            <label className="slider-label">
              <span>Coronal Cut (Y)</span>
              <strong>{sliceY > 0 ? `+${sliceY}` : sliceY} mm</strong>
            </label>
            <input
              type="range"
              min="-28"
              max="28"
              step="1"
              value={sliceY}
              onChange={(e) => setSliceY(parseInt(e.target.value))}
              className="custom-range-slider"
            />
          </div>
        </div>
      </div>
    </div>
  );
}
