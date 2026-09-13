/**
 * BigEarthNet 43-Class and 19-Class Standardized Taxonomy for SatQuery AI
 * Adapted for remote-sensing vision-language alignment.
 */
export const BIGEARTHNET_19_CLASSES = [
  'Urban fabric',
  'Industrial or commercial units',
  'Arable land',
  'Permanent crops',
  'Pastures',
  'Complex cultivation patterns',
  'Land principally agriculture with natural vegetation',
  'Agro-forestry areas',
  'Broad-leaved forest',
  'Coniferous forest',
  'Mixed forest',
  'Natural grassland & sparsely vegetated',
  'Moors, heathland & sclerophyllous vegetation',
  'Transitional woodland & shrub',
  'Beaches, dunes & sands',
  'Inland wetlands',
  'Coastal wetlands',
  'Inland waters',
  'Marine waters',
]

export const BENCHMARK_METRICS = [
  {
    benchmark: 'RSVQA',
    task: 'Single-Image Visual Question Answering',
    metrics: 'Top-1 Accuracy, Macro F1',
    dataset_split: 'Test Subset (High & Low Resolution splits)',
    target: 'Land cover, presence, counting, comparison',
  },
  {
    benchmark: 'VRSBench',
    task: 'Scene Captioning & Text-Guided Grounding',
    metrics: 'BLEU-4, CIDEr, ROUGE-L, Grounding mIoU@0.5',
    dataset_split: 'VRSBench Standard Test Split',
    target: 'Multi-attribute description & spatial coordinates',
  },
  {
    benchmark: 'CDVQA',
    task: 'Bi-Temporal Change Question Answering',
    metrics: 'Change Identification Acc, Attribution F1',
    dataset_split: 'CDVQA Bi-Temporal Test Set',
    target: 'Multi-date surface alteration & area expansion',
  },
  {
    benchmark: 'ISRO / SAC Evaluation Set',
    task: 'Optical–SAR Cross-Modal Joint Analysis',
    metrics: 'SSIM, Cross-Correlation, Joint Mask Complementarity',
    dataset_split: 'Cartosat-2S (Optical) + RISAT-1A (SAR) Pairs',
    target: 'Cross-sensor canopy penetration & structural identification',
  },
]
