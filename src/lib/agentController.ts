// SatQuery AI — Client-Side Agentic Controller & Types
// Shared with backend for consistent typing, routing, and execution trace visualization.

export type TaskType =
  | 'vqa'
  | 'caption'
  | 'grounding'
  | 'change_detection'
  | 'sar_optical_fusion'
  | 'building_detection'
  | 'vegetation_analysis'
  | 'flood_assessment'
  | 'unknown'

export interface ToolSpec {
  id: string
  name: string
  description: string
  supported_tasks: TaskType[]
  modalities: ('optical' | 'sar' | 'multispectral')[]
  adapter: string
  domain_adaptation: string
  model_id: string
  permitted_parameters?: Record<string, string | number | boolean>
}

export const TOOL_REGISTRY: Record<string, ToolSpec> = {
  rs_vqa: {
    id: 'rs_vqa',
    name: 'Remote-Sensing VQA Engine',
    description: 'Visual question answering adapted for satellite and aerial imagery using domain system prompts calibrated with BigEarthNet land-cover taxonomy.',
    supported_tasks: ['vqa', 'vegetation_analysis', 'flood_assessment', 'unknown'],
    modalities: ['optical', 'multispectral'],
    adapter: 'RS-Domain System Prompt (BigEarthNet 43-class taxonomy + RSVQA conventions)',
    domain_adaptation: 'BigEarthNet land-cover vocabulary mapping, sensor-specific spatial calibration, and visual confidence estimation.',
    model_id: 'claude-sonnet-5',
    permitted_parameters: {
      confidence_threshold: 0.75,
      domain_taxonomy: 'BigEarthNet-43',
      benchmark: 'RSVQA',
      max_tokens: 700,
    },
  },
  rs_building_detector: {
    id: 'rs_building_detector',
    name: 'Tiled YOLO Building Footprint Specialist',
    description: 'Deep learning instance segmentation model (building_model.pt) with tile coordinate mapping and polygon IoU duplicate removal for individual structural footprint auditing.',
    supported_tasks: ['building_detection'],
    modalities: ['optical'],
    adapter: 'PyTorch YOLO Segmentation Engine (Tiled Inference + NMS/IoU Deduplication)',
    domain_adaptation: 'Weight-trained on high-resolution aerial and satellite imagery for structural rooftop footprints with confidence calibration.',
    model_id: 'yolo-segmentation-building_model.pt',
    permitted_parameters: {
      tile_size_px: 512,
      overlap_px: 64,
      iou_threshold: 0.45,
      model: 'YOLOv8-Seg',
    },
  },
  rs_change_detector: {
    id: 'rs_change_detector',
    name: 'Bi-Temporal Change Detection Model',
    description: 'Analyses co-registered multi-temporal optical or SAR image pairs for structural and environmental change detection and CDVQA.',
    supported_tasks: ['change_detection'],
    modalities: ['optical', 'multispectral', 'sar'],
    adapter: 'RS-Change System Prompt (CDVQA benchmark conventions)',
    domain_adaptation: 'Calibrated for multi-temporal change identification, confidence reporting, and spatial change region bounding.',
    model_id: 'claude-sonnet-5',
    permitted_parameters: {
      coregistration_tolerance_px: 2.0,
      change_attribution: 'bi-temporal',
      benchmark: 'CDVQA',
      max_tokens: 800,
    },
  },
  rs_fusion_cv: {
    id: 'rs_fusion_cv',
    name: 'Optical–SAR Classical-CV Feature Extractor',
    description: 'Computes multi-modal telemetry: optical NDVI proxy, water/built-up indices, SAR backscatter intensity, speckle index, SSIM structural similarity, and cross-correlation.',
    supported_tasks: ['sar_optical_fusion'],
    modalities: ['optical', 'sar'],
    adapter: 'OpenCV / NumPy Classical Computer Vision Pipeline',
    domain_adaptation: 'Sensor-specific radar backscatter (dB), speckle noise modeling, and optical-SAR complementarity scoring.',
    model_id: 'classical-cv-fusion-engine-v2',
    permitted_parameters: {
      optical_sensor: 'Cartosat-2S',
      sar_sensor: 'RISAT-1A',
      decomposition: 'SSIM+CrossCorr',
    },
  },
  rs_captioner: {
    id: 'rs_captioner',
    name: 'RS Scene Captioning Engine',
    description: 'Generates structured scene-level captions covering land-cover types, dominant objects, spatial layout, and spectral characteristics per VRSBench standards.',
    supported_tasks: ['caption'],
    modalities: ['optical', 'multispectral', 'sar'],
    adapter: 'RS-Captioning System Prompt (VRSBench conventions)',
    domain_adaptation: 'Multi-attribute remote sensing description covering topography, land-use distribution, and sensor properties.',
    model_id: 'claude-sonnet-5',
    permitted_parameters: {
      caption_detail: 'multi-attribute',
      vocabulary: 'BigEarthNet-43',
      benchmark: 'VRSBench',
    },
  },
  rs_grounding: {
    id: 'rs_grounding',
    name: 'Text-Guided Spatial Grounding Engine',
    description: 'Identifies and localizes requested geographical objects or terrain patches, returning percentage-based bounding coordinates.',
    supported_tasks: ['grounding'],
    modalities: ['optical', 'multispectral'],
    adapter: 'RS-Grounding System Prompt (RSVQA / VRSBench grounding conventions)',
    domain_adaptation: 'Spatial coordinate bounding box prediction normalized to image frame dimensions.',
    model_id: 'claude-sonnet-5',
    permitted_parameters: {
      coordinate_system: 'normalized_percentage',
      bbox_format: '[x,y,w,h]',
      benchmark: 'VRSBench-Grounding',
    },
  },
}

export interface ExecutionTraceStep {
  step: number
  tool: string
  description: string
  input_summary: string
  output_summary: string
  duration_ms: number
  status: 'success' | 'skipped' | 'error'
  success?: boolean
  confidence_source?: 'real_inference' | 'heuristic' | 'none'
  parameters?: Record<string, string | number | boolean>
}

export interface InputValidation {
  images_provided: number
  modalities: string[]
  formats: string[]
  compatibility: 'valid' | 'warning' | 'error'
  notes: string[]
}

export interface ExecutionTrace {
  agent_version: string
  task_type: TaskType
  tools_invoked: string[]
  steps: ExecutionTraceStep[]
  total_duration_ms: number
  input_validation: InputValidation
  model_registry_entry: {
    model_id: string
    adapter: string
    domain_adaptation: string
    permitted_parameters?: Record<string, string | number | boolean>
  }
}

export interface FusionFeatures {
  optical: {
    vegetation_fraction: number
    water_fraction: number
    built_up_fraction: number
    spectral_entropy: number
  }
  sar: {
    mean_backscatter_db: number
    backscatter_std_db: number
    speckle_index: number
    edge_density: number
    roughness_fraction: number
  }
  cross_modal: {
    structural_similarity: number
    cross_correlation: number
    complementarity_index: number
    fusion_confidence: 'high' | 'medium' | 'low'
  }
}

export function classifyTask(
  question: string,
  imageCount = 1,
  modalities: ('optical' | 'sar' | 'multispectral')[] = ['optical']
): TaskType {
  const q = question.toLowerCase().trim()

  // 1. Dual-image or SAR+Optical explicitly requested
  if (
    modalities.includes('sar') ||
    q.includes('sar') ||
    q.includes('radar') ||
    q.includes('fusion') ||
    q.includes('cross-modal') ||
    q.includes('joint analysis') ||
    q.includes('optical and sar') ||
    q.includes('sar and optical') ||
    q.includes('sentinel-1') ||
    q.includes('risat')
  ) {
    if (imageCount >= 2 || modalities.length >= 2 || q.includes('optical and sar') || q.includes('sar and optical') || q.includes('identify built-up and water-covered')) {
      return 'sar_optical_fusion'
    }
  }

  // 2. Multi-temporal change detection (CDVQA)
  if (
    imageCount >= 2 ||
    q.includes('change') ||
    q.includes('compare') ||
    q.includes('before') ||
    q.includes('after') ||
    q.includes('difference') ||
    q.includes('temporal') ||
    q.includes('evolution') ||
    q.includes('bi-temporal') ||
    q.includes('expansion between') ||
    q.includes('increased') ||
    q.includes('decreased') ||
    q.includes('remained unchanged') ||
    q.includes('between these two dates')
  ) {
    return 'change_detection'
  }

  // 3. Grounding (Spatial localization - RSVQA / VRSBench Grounding)
  if (
    q.includes('locate') ||
    q.includes('find the') ||
    q.includes('where is') ||
    q.includes('pinpoint') ||
    q.includes('bounding box') ||
    q.includes('highlight') ||
    q.includes('demarcate') ||
    q.includes('water body referred to') ||
    q.includes('target area')
  ) {
    return 'grounding'
  }

  // 4. Scene Captioning (VRSBench)
  if (
    q.includes('caption') ||
    q.includes('describe') ||
    q.includes('scene overview') ||
    q.includes('detailed description') ||
    q.includes('land-cover and major objects') ||
    q.includes('major objects visible') ||
    q.includes('vrsbench')
  ) {
    return 'caption'
  }

  // 5. Building / Structure Detection
  if (
    q.includes('building') ||
    q.includes('footprint') ||
    q.includes('rooftop') ||
    q.includes('structure') ||
    q.includes('house') ||
    q.includes('how many buildings') ||
    q.includes('count buildings') ||
    q.includes('audit buildings') ||
    (q.includes('count') && (q.includes('urban') || q.includes('roof') || q.includes('built')))
  ) {
    return 'building_detection'
  }

  // 6. Vegetation / Crop Health
  if (
    q.includes('vegetation') ||
    q.includes('crop') ||
    q.includes('ndvi') ||
    q.includes('harvest') ||
    q.includes('canopy') ||
    q.includes('plant') ||
    q.includes('drought') ||
    q.includes('chlorophyll') ||
    q.includes('agriculture') ||
    q.includes('field health')
  ) {
    return 'vegetation_analysis'
  }

  // 7. Flood / Inundation
  if (
    q.includes('flood') ||
    q.includes('inundat') ||
    q.includes('water body') ||
    q.includes('submerge') ||
    q.includes('overflow') ||
    q.includes('lake') ||
    q.includes('river level')
  ) {
    return 'flood_assessment'
  }

  // Default to general Remote Sensing VQA
  return 'vqa'
}

export function formatTaskLabel(task: TaskType): string {
  switch (task) {
    case 'sar_optical_fusion':
      return 'Optical–SAR Cross-Modal Fusion'
    case 'building_detection':
      return 'YOLO Structural Footprint Audit'
    case 'change_detection':
      return 'Bi-Temporal Change Analysis (CDVQA)'
    case 'caption':
      return 'VRSBench Scene Captioning'
    case 'grounding':
      return 'Text-Guided Spatial Grounding'
    case 'vegetation_analysis':
      return 'Canopy & Agricultural NDVI Analysis'
    case 'flood_assessment':
      return 'Hydrological & Inundation Mapping'
    default:
      return 'Remote-Sensing VQA (RSVQA)'
  }
}
