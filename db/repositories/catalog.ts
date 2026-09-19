/**
 * db/repositories/catalog.ts — Dataset & model catalog queries (Phase 4)
 *
 * Read-only access to the dataset and model registry seeded by migration 002.
 * All datasets start as not_downloaded; all models start as unavailable.
 * Falls back to static in-memory catalog when DB is unavailable so the
 * /api/catalog/* endpoints always return meaningful data.
 */
import { dbQuery } from '../pool.js'

// ─── Types ────────────────────────────────────────────────────────────────────

export interface DatasetRecord {
  id: string
  name: string
  description: string | null
  source_url: string | null
  license: string | null
  modality: string
  version: string | null
  storage_uri: string | null
  availability_status: 'not_downloaded' | 'available'
  task_support: string[]
}

export interface ModelRecord {
  id: string
  name: string
  version: string | null
  enabled: boolean
  checkpoint_uri: string | null
  remote_sensing_adapted: boolean
  required_dataset: string | null
  availability: 'available' | 'unavailable'
  unavailable_reason: string | null
  capabilities: string[]
}

// ─── Static fallback catalog (used when DB is unavailable) ────────────────────
// Mirrors what migration 002 seeds.

const STATIC_DATASETS: DatasetRecord[] = [
  { id: 'bigearthnet_txt',  name: 'BigEarthNet.txt',    description: 'BigEarthNet multi-label annotation text files (43-class → 19-class CLC).', source_url: 'https://bigearth.net', license: 'CDLA-Permissive', modality: 'multispectral', version: '1.0', storage_uri: 'data/bigearthnet_txt', availability_status: 'not_downloaded', task_support: ['vqa'] },
  { id: 'bigearthnet_v2',   name: 'BigEarthNet v2.0',   description: 'reBEN — revised multi-label annotations, 19-class Corine taxonomy. Clasen et al., IGARSS 2025.', source_url: 'https://bigearth.net', license: 'CDLA-Permissive', modality: 'multispectral', version: '2.0', storage_uri: 'data/bigearthnet_v2', availability_status: 'not_downloaded', task_support: ['vqa'] },
  { id: 'vrsbench',         name: 'VRSBench',            description: 'Visual RS Benchmark — captioning, grounding, counting.', source_url: 'https://github.com/lx709/VRSBench', license: 'CC BY 4.0', modality: 'optical', version: '1.0', storage_uri: 'data/vrsbench', availability_status: 'not_downloaded', task_support: ['caption', 'grounding'] },
  { id: 'rsvqa_lr',         name: 'RSVQA-LR',            description: 'RS Visual QA — Low Resolution (Sentinel-2).', source_url: 'https://rsvqa.sylvainlobry.com', license: 'CC BY 4.0', modality: 'optical', version: '1.0', storage_uri: 'data/rsvqa_lr', availability_status: 'not_downloaded', task_support: ['vqa'] },
  { id: 'rsvqa_hr',         name: 'RSVQA-HR',            description: 'RS Visual QA — High Resolution (USGS aerial).', source_url: 'https://rsvqa.sylvainlobry.com', license: 'CC BY 4.0', modality: 'optical', version: '1.0', storage_uri: 'data/rsvqa_hr', availability_status: 'not_downloaded', task_support: ['vqa'] },
  { id: 'rsvqaxben',        name: 'RSVQAxBEN',           description: 'Cross-modal RSVQA over BigEarthNet imagery.', source_url: 'https://rsvqa.sylvainlobry.com', license: 'CC BY 4.0', modality: 'multispectral', version: '1.0', storage_uri: 'data/rsvqaxben', availability_status: 'not_downloaded', task_support: ['vqa'] },
  { id: 'cdvqa',            name: 'CDVQA',               description: 'Change Detection VQA — multi-temporal optical pairs.', source_url: 'https://github.com/YZHJenny/CDVQA', license: 'CC BY 4.0', modality: 'optical', version: '1.0', storage_uri: 'data/cdvqa', availability_status: 'not_downloaded', task_support: ['change_detection'] },
  { id: 'levir_cc',         name: 'LEVIR-CC',            description: 'LEVIR Change Captioning — bi-temporal satellite pairs with NL change descriptions.', source_url: 'https://github.com/Chen-Yang-Liu/LEVIR-CC-Dataset', license: 'CC BY-NC-SA 4.0', modality: 'optical', version: '1.0', storage_uri: 'data/levir_cc', availability_status: 'not_downloaded', task_support: ['change_detection'] },
  { id: 'levir_mci',        name: 'LEVIR-MCI',           description: 'LEVIR Multi-modal Change Inference with detailed attribute annotations.', source_url: 'https://github.com/Chen-Yang-Liu/LEVIR-CC-Dataset', license: 'CC BY-NC-SA 4.0', modality: 'optical', version: '1.0', storage_uri: 'data/levir_mci', availability_status: 'not_downloaded', task_support: ['change_detection'] },
  { id: 'vrsbench_sar',     name: 'VRSBench-SAR',        description: 'SAR extension of VRSBench — captioning and grounding for SAR imagery.', source_url: 'https://github.com/lx709/VRSBench', license: 'CC BY 4.0', modality: 'sar', version: '1.0', storage_uri: 'data/vrsbench_sar', availability_status: 'not_downloaded', task_support: ['sar_optical_fusion'] },
]

const STATIC_MODELS: ModelRecord[] = [
  { id: 'rs_vqa_ben',            name: 'RS-VQA (BigEarthNet adapted)',    version: '1.0', enabled: false, checkpoint_uri: null, remote_sensing_adapted: false, required_dataset: 'bigearthnet_v2', availability: 'unavailable', unavailable_reason: 'BigEarthNet v2.0 not downloaded; domain adaptation not performed.', capabilities: ['vqa'] },
  { id: 'rs_grounding_vrsbench', name: 'RS Grounding (VRSBench)',          version: '1.0', enabled: false, checkpoint_uri: null, remote_sensing_adapted: false, required_dataset: 'vrsbench', availability: 'unavailable', unavailable_reason: 'VRSBench not downloaded; grounding model not adapted.', capabilities: ['grounding'] },
  { id: 'rs_captioner_vrsbench', name: 'RS Scene Captioning (VRSBench)',   version: '1.0', enabled: false, checkpoint_uri: null, remote_sensing_adapted: false, required_dataset: 'vrsbench', availability: 'unavailable', unavailable_reason: 'VRSBench not downloaded; captioning model not adapted.', capabilities: ['caption'] },
  { id: 'rs_change_cdvqa',       name: 'Change-VQA (CDVQA)',               version: '1.0', enabled: false, checkpoint_uri: null, remote_sensing_adapted: false, required_dataset: 'cdvqa', availability: 'unavailable', unavailable_reason: 'CDVQA not downloaded; change-VQA model not adapted.', capabilities: ['change_detection'] },
  { id: 'rs_change_levir',       name: 'Change Detection (LEVIR)',          version: '1.0', enabled: false, checkpoint_uri: null, remote_sensing_adapted: false, required_dataset: 'levir_cc', availability: 'unavailable', unavailable_reason: 'LEVIR-CC not downloaded; change detection model not adapted.', capabilities: ['change_detection'] },
  { id: 'rs_sar_optical_fusion', name: 'Optical-SAR Fusion (VRSBench-SAR)',version: '1.0', enabled: false, checkpoint_uri: null, remote_sensing_adapted: false, required_dataset: 'vrsbench_sar', availability: 'unavailable', unavailable_reason: 'VRSBench-SAR not downloaded; fusion model not adapted.', capabilities: ['sar_optical_fusion'] },
  { id: 'rs_building_yolo',      name: 'Building Detection (YOLO)',         version: '1.0', enabled: false, checkpoint_uri: 'backend/models/building_model.pt', remote_sensing_adapted: false, required_dataset: null, availability: 'unavailable', unavailable_reason: 'YOLO checkpoint (building_model.pt) not installed.', capabilities: ['building_detection'] },
]

// ─── Dataset queries ──────────────────────────────────────────────────────────

/**
 * Returns all dataset records with task support arrays.
 * Falls back to STATIC_DATASETS when DB is unavailable.
 */
export async function listDatasets(): Promise<DatasetRecord[]> {
  const result = await dbQuery<{
    id: string; name: string; description: string | null; source_url: string | null;
    license: string | null; modality: string; version: string | null;
    storage_uri: string | null; availability_status: string;
    tasks: string | null;
  }>(`
    SELECT d.*,
           STRING_AGG(dts.task_type, ',') AS tasks
    FROM datasets d
    LEFT JOIN dataset_task_support dts ON dts.dataset_id = d.id
    GROUP BY d.id
    ORDER BY d.name
  `)

  if (!result || result.rowCount === 0) return STATIC_DATASETS

  return result.rows.map((r) => ({
    id: r.id,
    name: r.name,
    description: r.description,
    source_url: r.source_url,
    license: r.license,
    modality: r.modality,
    version: r.version,
    storage_uri: r.storage_uri,
    availability_status: r.availability_status as 'not_downloaded' | 'available',
    task_support: r.tasks ? r.tasks.split(',') : [],
  }))
}

/**
 * Returns catalog statistics: total, available, not_downloaded counts.
 */
export async function datasetStats(): Promise<{ total: number; available: number; not_downloaded: number }> {
  const result = await dbQuery<{ total: string; available: string; not_downloaded: string }>(`
    SELECT
      COUNT(*) AS total,
      COUNT(*) FILTER (WHERE availability_status = 'available') AS available,
      COUNT(*) FILTER (WHERE availability_status = 'not_downloaded') AS not_downloaded
    FROM datasets
  `)
  if (!result || result.rowCount === 0) {
    return { total: STATIC_DATASETS.length, available: 0, not_downloaded: STATIC_DATASETS.length }
  }
  const r = result.rows[0]
  return { total: Number(r.total), available: Number(r.available), not_downloaded: Number(r.not_downloaded) }
}

// ─── Model queries ────────────────────────────────────────────────────────────

/**
 * Returns all model records with capability arrays.
 * Falls back to STATIC_MODELS when DB is unavailable.
 */
export async function listModels(): Promise<ModelRecord[]> {
  const result = await dbQuery<{
    id: string; name: string; version: string | null; enabled: boolean;
    checkpoint_uri: string | null; remote_sensing_adapted: boolean;
    required_dataset: string | null; availability: string;
    unavailable_reason: string | null; caps: string | null;
  }>(`
    SELECT m.*,
           STRING_AGG(mc.task_type, ',') AS caps
    FROM models m
    LEFT JOIN model_capabilities mc ON mc.model_id = m.id
    GROUP BY m.id
    ORDER BY m.name
  `)

  if (!result || result.rowCount === 0) return STATIC_MODELS

  return result.rows.map((r) => ({
    id: r.id,
    name: r.name,
    version: r.version,
    enabled: r.enabled,
    checkpoint_uri: r.checkpoint_uri,
    remote_sensing_adapted: r.remote_sensing_adapted,
    required_dataset: r.required_dataset,
    availability: r.availability as 'available' | 'unavailable',
    unavailable_reason: r.unavailable_reason,
    capabilities: r.caps ? r.caps.split(',') : [],
  }))
}

/**
 * Returns model registry statistics.
 */
export async function modelStats(): Promise<{ total: number; available: number; unavailable: number }> {
  const result = await dbQuery<{ total: string; available: string; unavailable: string }>(`
    SELECT
      COUNT(*) AS total,
      COUNT(*) FILTER (WHERE availability = 'available') AS available,
      COUNT(*) FILTER (WHERE availability = 'unavailable') AS unavailable
    FROM models
  `)
  if (!result || result.rowCount === 0) {
    return { total: STATIC_MODELS.length, available: 0, unavailable: STATIC_MODELS.length }
  }
  const r = result.rows[0]
  return { total: Number(r.total), available: Number(r.available), unavailable: Number(r.unavailable) }
}
