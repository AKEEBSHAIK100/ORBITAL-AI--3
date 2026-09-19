-- db/migrations/002_seed_catalog.sql
-- Phase 4: Seed remote-sensing dataset catalog and model registry
-- ALL datasets are marked not_downloaded — no files required at startup
-- ALL models are marked unavailable — no checkpoints installed
-- Idempotent: uses ON CONFLICT DO NOTHING

-- ─── Dataset Catalog ───────────────────────────────────────────────────────────

INSERT INTO datasets (id, name, description, source_url, license, modality, version, storage_uri, availability_status)
VALUES
  (
    'bigearthnet_txt',
    'BigEarthNet.txt',
    'BigEarthNet multi-label land-cover annotation text files (Corine Land Cover taxonomy, 43-class → 19-class).',
    'https://bigearth.net',
    'Community Data License Agreement — Permissive',
    'multispectral',
    '1.0',
    'data/bigearthnet_txt',
    'not_downloaded'
  ),
  (
    'bigearthnet_v2',
    'BigEarthNet v2.0',
    'reBEN — revised BigEarthNet with corrected multi-label annotations. 19-class Corine taxonomy. Sentinel-2 patches. Clasen et al., IGARSS 2025.',
    'https://bigearth.net',
    'Community Data License Agreement — Permissive',
    'multispectral',
    '2.0',
    'data/bigearthnet_v2',
    'not_downloaded'
  ),
  (
    'vrsbench',
    'VRSBench',
    'Visual Remote Sensing Benchmark for scene captioning, object counting, and spatial grounding tasks.',
    'https://github.com/lx709/VRSBench',
    'CC BY 4.0',
    'optical',
    '1.0',
    'data/vrsbench',
    'not_downloaded'
  ),
  (
    'rsvqa_lr',
    'RSVQA-LR',
    'Remote Sensing Visual Question Answering — Low Resolution (Sentinel-2 derived imagery).',
    'https://rsvqa.sylvainlobry.com',
    'CC BY 4.0',
    'optical',
    '1.0',
    'data/rsvqa_lr',
    'not_downloaded'
  ),
  (
    'rsvqa_hr',
    'RSVQA-HR',
    'Remote Sensing Visual Question Answering — High Resolution (USGS aerial / aerial orthophotos).',
    'https://rsvqa.sylvainlobry.com',
    'CC BY 4.0',
    'optical',
    '1.0',
    'data/rsvqa_hr',
    'not_downloaded'
  ),
  (
    'rsvqaxben',
    'RSVQAxBEN',
    'Cross-modal RSVQA over BigEarthNet imagery, combining VQA labels with multi-label land cover annotations.',
    'https://rsvqa.sylvainlobry.com',
    'CC BY 4.0',
    'multispectral',
    '1.0',
    'data/rsvqaxben',
    'not_downloaded'
  ),
  (
    'cdvqa',
    'CDVQA',
    'Change Detection Visual Question Answering — multi-temporal optical image pairs with change-oriented QA annotations.',
    'https://github.com/YZHJenny/CDVQA',
    'CC BY 4.0',
    'optical',
    '1.0',
    'data/cdvqa',
    'not_downloaded'
  ),
  (
    'levir_cc',
    'LEVIR-CC',
    'LEVIR Change Captioning dataset — high-resolution bi-temporal satellite image pairs with natural-language change descriptions.',
    'https://github.com/Chen-Yang-Liu/LEVIR-CC-Dataset',
    'CC BY-NC-SA 4.0',
    'optical',
    '1.0',
    'data/levir_cc',
    'not_downloaded'
  ),
  (
    'levir_mci',
    'LEVIR-MCI',
    'LEVIR Multi-modal Change Inference — extends LEVIR-CC with detailed change attribute annotations.',
    'https://github.com/Chen-Yang-Liu/LEVIR-CC-Dataset',
    'CC BY-NC-SA 4.0',
    'optical',
    '1.0',
    'data/levir_mci',
    'not_downloaded'
  ),
  (
    'vrsbench_sar',
    'VRSBench-SAR',
    'SAR extension of VRSBench — synthetic aperture radar imagery with captioning and grounding annotations.',
    'https://github.com/lx709/VRSBench',
    'CC BY 4.0',
    'sar',
    '1.0',
    'data/vrsbench_sar',
    'not_downloaded'
  )
ON CONFLICT (id) DO NOTHING;

-- ─── Dataset Task Support ──────────────────────────────────────────────────────

INSERT INTO dataset_task_support (dataset_id, task_type, benchmark_name)
VALUES
  ('bigearthnet_v2',  'vqa',              'BigEarthNet-19'),
  ('bigearthnet_txt', 'vqa',              'BigEarthNet-19'),
  ('rsvqa_lr',        'vqa',              'RSVQA-LR'),
  ('rsvqa_hr',        'vqa',              'RSVQA-HR'),
  ('rsvqaxben',       'vqa',              'RSVQAxBEN'),
  ('vrsbench',        'caption',          'VRSBench-Caption'),
  ('vrsbench',        'grounding',        'VRSBench-Grounding'),
  ('vrsbench_sar',    'sar_optical_fusion','VRSBench-SAR'),
  ('cdvqa',           'change_detection', 'CDVQA'),
  ('levir_cc',        'change_detection', 'LEVIR-CC'),
  ('levir_mci',       'change_detection', 'LEVIR-MCI')
ON CONFLICT (dataset_id, task_type) DO NOTHING;

-- ─── Model Registry ────────────────────────────────────────────────────────────
-- All models unavailable because no checkpoints are installed.
-- required_dataset points to the dataset needed for domain adaptation.

INSERT INTO models (id, name, version, enabled, checkpoint_uri, remote_sensing_adapted, required_dataset, availability, unavailable_reason)
VALUES
  (
    'rs_vqa_ben',
    'RS-VQA (BigEarthNet adapted)',
    '1.0',
    FALSE,
    NULL,
    FALSE,
    'bigearthnet_v2',
    'unavailable',
    'BigEarthNet v2.0 dataset not downloaded; domain adaptation not performed.'
  ),
  (
    'rs_grounding_vrsbench',
    'RS Grounding (VRSBench)',
    '1.0',
    FALSE,
    NULL,
    FALSE,
    'vrsbench',
    'unavailable',
    'VRSBench dataset not downloaded; spatial grounding model not adapted.'
  ),
  (
    'rs_captioner_vrsbench',
    'RS Scene Captioning (VRSBench)',
    '1.0',
    FALSE,
    NULL,
    FALSE,
    'vrsbench',
    'unavailable',
    'VRSBench dataset not downloaded; captioning model not adapted.'
  ),
  (
    'rs_change_cdvqa',
    'Change-VQA (CDVQA)',
    '1.0',
    FALSE,
    NULL,
    FALSE,
    'cdvqa',
    'unavailable',
    'CDVQA dataset not downloaded; change-VQA model not adapted.'
  ),
  (
    'rs_change_levir',
    'Change Detection (LEVIR)',
    '1.0',
    FALSE,
    NULL,
    FALSE,
    'levir_cc',
    'unavailable',
    'LEVIR-CC dataset not downloaded; change detection model not adapted.'
  ),
  (
    'rs_sar_optical_fusion',
    'Optical-SAR Fusion (VRSBench-SAR)',
    '1.0',
    FALSE,
    NULL,
    FALSE,
    'vrsbench_sar',
    'unavailable',
    'VRSBench-SAR dataset not downloaded; fusion model not adapted.'
  ),
  (
    'rs_building_yolo',
    'Building Detection (YOLO)',
    '1.0',
    FALSE,
    'backend/models/building_model.pt',
    FALSE,
    NULL,
    'unavailable',
    'YOLO checkpoint (building_model.pt) not installed.'
  )
ON CONFLICT (id) DO NOTHING;

-- ─── Model Capabilities (declared, not measured — awaiting adaptation) ─────────

INSERT INTO model_capabilities (model_id, task_type, benchmark, notes)
VALUES
  ('rs_vqa_ben',           'vqa',              'RSVQA-LR/HR',     'Pending dataset download and fine-tuning.'),
  ('rs_grounding_vrsbench','grounding',         'VRSBench',        'Pending dataset download and fine-tuning.'),
  ('rs_captioner_vrsbench','caption',           'VRSBench',        'Pending dataset download and fine-tuning.'),
  ('rs_change_cdvqa',      'change_detection',  'CDVQA',           'Pending dataset download and fine-tuning.'),
  ('rs_change_levir',      'change_detection',  'LEVIR-CC',        'Pending dataset download and fine-tuning.'),
  ('rs_sar_optical_fusion','sar_optical_fusion','VRSBench-SAR',    'Pending dataset download and fine-tuning.'),
  ('rs_building_yolo',     'building_detection','Aerial/Sat Aerial','Pending checkpoint installation.')
ON CONFLICT (model_id, task_type) DO NOTHING;
