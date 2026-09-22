"""
SatQuery AI — Remote-Sensing VLM Fixed 40-Question Evaluation Suite.

Provides 40 deterministic evaluation questions across exactly 20 remote-sensing categories
(2 questions per category).
Uses real satellite imagery present in the workspace.
Never fabricates ground truth answers.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

from .schemas import EvaluationQuestion, ImageSample

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_evaluation_samples() -> Dict[str, ImageSample]:
    """Resolves local verified satellite imagery samples."""
    default_aerial = PROJECT_ROOT / "backend" / "data" / "default_aerial.jpg"
    levir_a = PROJECT_ROOT / "backend" / "data" / "levir_cd" / "A" / "test_2_0000_0000.png"
    levir_b = PROJECT_ROOT / "backend" / "data" / "levir_cd" / "B" / "test_2_0000_0000.png"
    opt_tif = PROJECT_ROOT / "backend" / "data" / "optical_sar" / "optical.tif"
    sar_tif = PROJECT_ROOT / "backend" / "data" / "optical_sar" / "sar.tif"

    return {
        "sample_optical_aerial": ImageSample(
            image_id="sample_optical_aerial",
            image_path=str(default_aerial.resolve()),
            dimensions=[2400, 1799] if default_aerial.exists() else None,
            modality="optical_rgb",
            crs=None,
            temporal_info="single_acquisition",
            is_paired=False,
            is_coregistered=None,
            metadata_verified=default_aerial.exists(),
        ),
        "sample_levir_cd_pair": ImageSample(
            image_id="sample_levir_cd_pair",
            image_path=str(levir_a.resolve()),
            secondary_image_path=str(levir_b.resolve()),
            dimensions=[256, 256] if levir_a.exists() else None,
            secondary_dimensions=[256, 256] if levir_b.exists() else None,
            modality="optical_rgb",
            secondary_modality="optical_rgb",
            crs=None,
            temporal_info="bi_temporal_t1_t2",
            is_paired=True,
            is_coregistered=True,
            metadata_verified=levir_a.exists() and levir_b.exists(),
        ),
        "sample_sar_single": ImageSample(
            image_id="sample_sar_single",
            image_path=str(sar_tif.resolve()),
            dimensions=[518, 531] if sar_tif.exists() else None,
            modality="sar",
            crs="EPSG:32643 (UTM 43N)",
            temporal_info="single_acquisition",
            is_paired=False,
            is_coregistered=None,
            metadata_verified=sar_tif.exists(),
        ),
        "sample_optical_sar_pair": ImageSample(
            image_id="sample_optical_sar_pair",
            image_path=str(opt_tif.resolve()),
            secondary_image_path=str(sar_tif.resolve()),
            dimensions=[536, 679] if opt_tif.exists() else None,
            secondary_dimensions=[518, 531] if sar_tif.exists() else None,
            modality="optical_geotiff",
            secondary_modality="sar_geotiff",
            crs="EPSG:32643 (UTM 43N)",
            temporal_info="cross_sensor_unaligned",
            is_paired=True,
            is_coregistered=False,  # Unaligned dimensions 679x536 vs 531x518
            metadata_verified=opt_tif.exists() and sar_tif.exists(),
        ),
    }


def get_40_question_suite() -> List[EvaluationQuestion]:
    """
    Returns the fixed 40-question deterministic benchmark suite.
    20 categories * 2 questions each = exactly 40 questions.
    Ground truth is set to None/unavailable unless verified.
    """
    return [
        # Category 1: Scene description
        EvaluationQuestion(
            question_id="RS-SCN-01",
            category="scene_description",
            category_name="Scene Description",
            question_text="Describe the overall scene and dominant landscape characteristics visible in this satellite image.",
            input_type="single",
            image_role="optical",
            sample_id="sample_optical_aerial",
        ),
        EvaluationQuestion(
            question_id="RS-SCN-02",
            category="scene_description",
            category_name="Scene Description",
            question_text="What are the primary natural and man-made elements visible across this scene?",
            input_type="single",
            image_role="optical",
            sample_id="sample_optical_aerial",
        ),
        # Category 2: Land cover
        EvaluationQuestion(
            question_id="RS-LND-01",
            category="land_cover",
            category_name="Land Cover",
            question_text="What types of land cover are visible in this satellite image?",
            input_type="single",
            image_role="optical",
            sample_id="sample_optical_aerial",
        ),
        EvaluationQuestion(
            question_id="RS-LND-02",
            category="land_cover",
            category_name="Land Cover",
            question_text="What is the dominant land-use or land-cover class across the observed area?",
            input_type="single",
            image_role="optical",
            sample_id="sample_optical_aerial",
        ),
        # Category 3: Vegetation
        EvaluationQuestion(
            question_id="RS-VEG-01",
            category="vegetation",
            category_name="Vegetation",
            question_text="Is vegetation present in this image, and if so, what type of canopy or ground vegetation is discernible?",
            input_type="single",
            image_role="optical",
            sample_id="sample_optical_aerial",
        ),
        EvaluationQuestion(
            question_id="RS-VEG-02",
            category="vegetation",
            category_name="Vegetation",
            question_text="Are there visible tree canopies, green spaces, or vegetated boundaries within the scene?",
            input_type="single",
            image_role="optical",
            sample_id="sample_optical_aerial",
        ),
        # Category 4: Water
        EvaluationQuestion(
            question_id="RS-WTR-01",
            category="water",
            category_name="Water",
            question_text="Is there any visible surface water, such as a river, canal, lake, or retention pond in this image?",
            input_type="single",
            image_role="optical",
            sample_id="sample_optical_aerial",
        ),
        EvaluationQuestion(
            question_id="RS-WTR-02",
            category="water",
            category_name="Water",
            question_text="Are drainage channels, water bodies, or wetlands observable in this satellite view?",
            input_type="single",
            image_role="optical",
            sample_id="sample_optical_aerial",
        ),
        # Category 5: Agriculture
        EvaluationQuestion(
            question_id="RS-AGR-01",
            category="agriculture",
            category_name="Agriculture",
            question_text="Does this scene appear agricultural, or are cultivated crop parcels and farming plots visible?",
            input_type="single",
            image_role="optical",
            sample_id="sample_optical_aerial",
        ),
        EvaluationQuestion(
            question_id="RS-AGR-02",
            category="agriculture",
            category_name="Agriculture",
            question_text="Are there regular geometric field boundaries, furrow patterns, or agricultural irrigation signatures?",
            input_type="single",
            image_role="optical",
            sample_id="sample_optical_aerial",
        ),
        # Category 6: Buildings
        EvaluationQuestion(
            question_id="RS-BLD-01",
            category="buildings",
            category_name="Buildings",
            question_text="Are buildings or residential structures visible in this scene?",
            input_type="single",
            image_role="optical",
            sample_id="sample_optical_aerial",
        ),
        EvaluationQuestion(
            question_id="RS-BLD-02",
            category="buildings",
            category_name="Buildings",
            question_text="What can you deduce about the density, layout, or roof structures of the buildings in this image?",
            input_type="single",
            image_role="optical",
            sample_id="sample_optical_aerial",
        ),
        # Category 7: Infrastructure
        EvaluationQuestion(
            question_id="RS-INF-01",
            category="infrastructure",
            category_name="Infrastructure",
            question_text="What civil, transportation, or utility infrastructure is visible in this scene?",
            input_type="single",
            image_role="optical",
            sample_id="sample_optical_aerial",
        ),
        EvaluationQuestion(
            question_id="RS-INF-02",
            category="infrastructure",
            category_name="Infrastructure",
            question_text="Are engineered structures, industrial compounds, or utility corridors present?",
            input_type="single",
            image_role="optical",
            sample_id="sample_optical_aerial",
        ),
        # Category 8: Roads
        EvaluationQuestion(
            question_id="RS-ROD-01",
            category="roads",
            category_name="Roads",
            question_text="Are roads, paved streets, or transport networks visible in this image?",
            input_type="single",
            image_role="optical",
            sample_id="sample_optical_aerial",
        ),
        EvaluationQuestion(
            question_id="RS-ROD-02",
            category="roads",
            category_name="Roads",
            question_text="Describe the roadway network pattern, intersections, or pathways visible in the scene.",
            input_type="single",
            image_role="optical",
            sample_id="sample_optical_aerial",
        ),
        # Category 9: Ships
        EvaluationQuestion(
            question_id="RS-SHP-01",
            category="ships",
            category_name="Ships",
            question_text="Are there any ships, maritime vessels, or boats visible in this scene?",
            input_type="single",
            image_role="optical",
            sample_id="sample_optical_aerial",
        ),
        EvaluationQuestion(
            question_id="RS-SHP-02",
            category="ships",
            category_name="Ships",
            question_text="Is there any harbor, dockside mooring, or marine traffic observable in the image?",
            input_type="single",
            image_role="optical",
            sample_id="sample_optical_aerial",
        ),
        # Category 10: Aircraft
        EvaluationQuestion(
            question_id="RS-AIR-01",
            category="aircraft",
            category_name="Aircraft",
            question_text="Are aircraft visible on the ground or in flight in this satellite image?",
            input_type="single",
            image_role="optical",
            sample_id="sample_optical_aerial",
        ),
        EvaluationQuestion(
            question_id="RS-AIR-02",
            category="aircraft",
            category_name="Aircraft",
            question_text="Are runways, taxiways, hangars, or airport aprons observable?",
            input_type="single",
            image_role="optical",
            sample_id="sample_optical_aerial",
        ),
        # Category 11: Vehicles
        EvaluationQuestion(
            question_id="RS-VEH-01",
            category="vehicles",
            category_name="Vehicles",
            question_text="Are vehicles (such as cars, trucks, or heavy machinery) visible on roads or parking lots?",
            input_type="single",
            image_role="optical",
            sample_id="sample_optical_aerial",
        ),
        EvaluationQuestion(
            question_id="RS-VEH-02",
            category="vehicles",
            category_name="Vehicles",
            question_text="Can individual ground transport vehicles or parking clusters be resolved at this image resolution?",
            input_type="single",
            image_role="optical",
            sample_id="sample_optical_aerial",
        ),
        # Category 12: Spatial location
        EvaluationQuestion(
            question_id="RS-LOC-01",
            category="spatial_location",
            category_name="Spatial Location",
            question_text="Where in the image (e.g., upper left, center, bottom right) are the most prominent structures located?",
            input_type="single",
            image_role="optical",
            sample_id="sample_optical_aerial",
        ),
        EvaluationQuestion(
            question_id="RS-LOC-02",
            category="spatial_location",
            category_name="Spatial Location",
            question_text="In which quadrants or sectors of the image is vegetation concentrated?",
            input_type="single",
            image_role="optical",
            sample_id="sample_optical_aerial",
        ),
        # Category 13: Spatial relationships
        EvaluationQuestion(
            question_id="RS-REL-01",
            category="spatial_relationships",
            category_name="Spatial Relationships",
            question_text="Where are the buildings located relative to the roadways or adjacent open spaces?",
            input_type="single",
            image_role="optical",
            sample_id="sample_optical_aerial",
        ),
        EvaluationQuestion(
            question_id="RS-REL-02",
            category="spatial_relationships",
            category_name="Spatial Relationships",
            question_text="Describe the spatial arrangement between built structures and surrounding natural ground cover.",
            input_type="single",
            image_role="optical",
            sample_id="sample_optical_aerial",
        ),
        # Category 14: Urban scene reasoning
        EvaluationQuestion(
            question_id="RS-URB-01",
            category="urban_scene_reasoning",
            category_name="Urban Scene Reasoning",
            question_text="Does this scene depict an urban, suburban, or planned human settlement?",
            input_type="single",
            image_role="optical",
            sample_id="sample_optical_aerial",
        ),
        EvaluationQuestion(
            question_id="RS-URB-02",
            category="urban_scene_reasoning",
            category_name="Urban Scene Reasoning",
            question_text="What indicators of human activity, zoning, or urban density can be inferred?",
            input_type="single",
            image_role="optical",
            sample_id="sample_optical_aerial",
        ),
        # Category 15: Rural scene reasoning
        EvaluationQuestion(
            question_id="RS-RUR-01",
            category="rural_scene_reasoning",
            category_name="Rural Scene Reasoning",
            question_text="Does this landscape exhibit rural, open-country, or undeveloped terrain characteristics?",
            input_type="single",
            image_role="optical",
            sample_id="sample_optical_aerial",
        ),
        EvaluationQuestion(
            question_id="RS-RUR-02",
            category="rural_scene_reasoning",
            category_name="Rural Scene Reasoning",
            question_text="What natural topographic features, unpaved surfaces, or rural plot divisions are visible?",
            input_type="single",
            image_role="optical",
            sample_id="sample_optical_aerial",
        ),
        # Category 16: Captioning
        EvaluationQuestion(
            question_id="RS-CAP-01",
            category="captioning",
            category_name="Captioning",
            question_text="Generate a concise, factual remote-sensing caption summarizing this overhead image.",
            input_type="single",
            image_role="optical",
            sample_id="sample_optical_aerial",
        ),
        EvaluationQuestion(
            question_id="RS-CAP-02",
            category="captioning",
            category_name="Captioning",
            question_text="Provide a formal satellite image descriptor documenting terrain, structures, and land use.",
            input_type="single",
            image_role="optical",
            sample_id="sample_optical_aerial",
        ),
        # Category 17: Grounding
        EvaluationQuestion(
            question_id="RS-GND-01",
            category="grounding",
            category_name="Visual Grounding",
            question_text="Identify and specify the image regions or bounding coordinates where prominent buildings reside.",
            input_type="single",
            image_role="optical",
            sample_id="sample_optical_aerial",
        ),
        EvaluationQuestion(
            question_id="RS-GND-02",
            category="grounding",
            category_name="Visual Grounding",
            question_text="Locate the principal roadway or access path across the scene by its spatial coordinates or quadrant.",
            input_type="single",
            image_role="optical",
            sample_id="sample_optical_aerial",
        ),
        # Category 18: SAR understanding
        EvaluationQuestion(
            question_id="RS-SAR-01",
            category="sar_understanding",
            category_name="SAR Understanding",
            question_text="In this radar/SAR image, what features exhibit high microwave backscatter (bright return) versus low backscatter (dark return)?",
            input_type="single",
            image_role="sar",
            sample_id="sample_sar_single",
        ),
        EvaluationQuestion(
            question_id="RS-SAR-02",
            category="sar_understanding",
            category_name="SAR Understanding",
            question_text="Comparing this optical and SAR pair, what complementary structural details or surface roughness characteristics are highlighted in radar?",
            input_type="pair",
            image_role="optical_sar",
            sample_id="sample_optical_sar_pair",
        ),
        # Category 19: Temporal/change reasoning
        EvaluationQuestion(
            question_id="RS-CHG-01",
            category="temporal_change_reasoning",
            category_name="Temporal Change Reasoning",
            question_text="What significant architectural, structural, or surface changes appear between Image A (T1) and Image B (T2)?",
            input_type="pair",
            image_role="temporal_pair",
            sample_id="sample_levir_cd_pair",
        ),
        EvaluationQuestion(
            question_id="RS-CHG-02",
            category="temporal_change_reasoning",
            category_name="Temporal Change Reasoning",
            question_text="Are there new buildings constructed or existing buildings removed between these two temporal acquisitions?",
            input_type="pair",
            image_role="temporal_pair",
            sample_id="sample_levir_cd_pair",
        ),
        # Category 20: Open-vocabulary RS reasoning
        EvaluationQuestion(
            question_id="RS-OPN-01",
            category="open_vocabulary_rs",
            category_name="Open-Vocabulary RS Reasoning",
            question_text="What broader environmental, industrial, or developmental inferences can be drawn from this satellite view?",
            input_type="single",
            image_role="optical",
            sample_id="sample_optical_aerial",
        ),
        EvaluationQuestion(
            question_id="RS-OPN-02",
            category="open_vocabulary_rs",
            category_name="Open-Vocabulary RS Reasoning",
            question_text="What potential environmental or operational risks (such as flood exposure or encroachment) can be hypothesized from this scene?",
            input_type="single",
            image_role="optical",
            sample_id="sample_optical_aerial",
        ),
    ]
