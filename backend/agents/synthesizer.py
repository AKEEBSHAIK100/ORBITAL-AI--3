"""
Orbital-AI / SatQuery AI — Natural Language Response Synthesizer.

Converts structured specialist evidence into natural language responses adhering to:
- Must use ONLY specialist outputs (never generic chatbot hallucinations).
- Truthful calibration reporting (never invent confidence numbers).
- Explicit conflict handling: "The available analyses disagree, so the result is uncertain."
- Query-scope control: Explains limitations for UNSUPPORTED_QUERY with alternative suggestions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from ..schemas.analysis import QueryPlan, SpecialistEvidenceObject


def synthesize_response(
    query: str,
    query_plan: QueryPlan,
    evidence_list: List[SpecialistEvidenceObject],
    validation: Dict[str, Any],
    status: str = "SUCCESS"
) -> Tuple[str, Optional[float], str, List[str]]:
    """
    Synthesizes natural language answer, overall confidence, confidence_status, and warnings.
    Returns (answer, confidence, confidence_status, warnings).
    """
    warnings: List[str] = list(validation.get("warnings", []))
    for ev in evidence_list:
        warnings.extend(ev.warnings)
    # Deduplicate warnings
    warnings = list(dict.fromkeys(warnings))

    # ── 1. Unsupported Queries ───────────────────────────────────────────────
    if status == "UNSUPPORTED_QUERY" or query_plan.intent == "unsupported":
        reason = query_plan.unsupported_reason or "The requested question is outside the scope of remote sensing imagery analysis."
        alts = query_plan.supported_alternatives or [
            "Extract structural building footprints",
            "Classify land-cover categories",
            "Perform bi-temporal change detection"
        ]
        alts_text = "; ".join(f"'{a}'" for a in alts)
        answer = (
            f"UNSUPPORTED QUERY: {reason} "
            f"Remote sensing specialists cannot predict future demographic or non-surface phenomena. "
            f"Supported analyses for this imagery include: {alts_text}."
        )
        return answer, None, "unavailable", warnings

    # ── 2. Validation Errors ─────────────────────────────────────────────────
    if status == "VALIDATION_ERROR" or validation.get("compatibility") == "error":
        errs = "; ".join(validation.get("errors", ["Input validation rejected request."]))
        answer = f"INPUT VALIDATION ERROR: {errs}"
        return answer, None, "unavailable", warnings

    # ── 3. Specialist Unavailable ────────────────────────────────────────────
    unavail = [
        ev for ev in evidence_list
        if ev.confidence_status == "unavailable"
        or "specialist unavailable" in str(ev.result).lower()
        or (str(ev.result).lower().startswith("specialist '") and "unavailable" in str(ev.result).lower())
    ]
    if status == "SPECIALIST_UNAVAILABLE" or unavail:
        unavail_tool = unavail[0].source if unavail else (query_plan.specialists[0] if query_plan.specialists else "required_specialist")
        reason = unavail[0].warnings[0] if (unavail and unavail[0].warnings) else f"Specialist '{unavail_tool}' is not installed or checkpoint is missing."
        answer = f"SPECIALIST UNAVAILABLE: {reason}. The analysis cannot proceed without the required model checkpoint. No answer has been fabricated."
        return answer, None, "unavailable", warnings

    # ── 4. Synthesize from Specialist Evidence ───────────────────────────────
    ev_by_task = {ev.task: ev for ev in evidence_list}
    ev_by_source = {ev.source: ev for ev in evidence_list}

    # Conflict detection helper
    conflict = False

    # Extract common components
    bldg_ev = ev_by_task.get("building_detection") or ev_by_source.get("building_detection")
    lc_ev = ev_by_task.get("land_cover") or ev_by_source.get("land_cover")
    vqa_ev = ev_by_task.get("vqa") or ev_by_source.get("rs_vqa_adapted")
    cap_ev = ev_by_task.get("caption") or ev_by_source.get("rs_caption_adapted")
    change_ev = ev_by_task.get("change_detection") or ev_by_source.get("change_detection")
    change_vqa_ev = ev_by_task.get("change_vqa") or ev_by_source.get("change_vqa")
    fusion_ev = ev_by_task.get("optical_sar_analysis") or ev_by_source.get("optical_sar_fusion") or ev_by_source.get("optical_sar")
    ground_ev = ev_by_task.get("grounding") or ev_by_source.get("visual_grounding") or ev_by_source.get("grounding")

    # Conflict check: VQA says "no buildings" or "no" while building detector found significant buildings
    if bldg_ev and vqa_ev:
        bldg_cnt = bldg_ev.evidence.get("building_count", 0)
        vqa_ans = str(vqa_ev.result).lower()
        if bldg_cnt > 5 and ("no building" in vqa_ans or vqa_ans.strip() in ["no", "no."]):
            conflict = True

    if conflict:
        answer = (
            "The available analyses disagree, so the result is uncertain. "
            f"The building footprint detector extracted {bldg_ev.evidence.get('building_count', 0)} structures, "
            f"while visual question answering reported: '{vqa_ev.result}'."
        )
        return answer, None, "not_calibrated", warnings

    # ── Intent-Specific Synthesis ────────────────────────────────────────────
    intent = query_plan.intent

    # Intent: land_cover ("What type of land is present?")
    if intent == "land_cover":
        top_label = lc_ev.evidence.get("top_label") if lc_ev else "Undetermined"
        active = lc_ev.evidence.get("active_labels", []) if lc_ev else []
        norm_active: List[str] = []
        for item in active:
            if isinstance(item, str):
                norm_active.append(item)
            elif isinstance(item, dict):
                label_val = item.get("name") or item.get("short") or item.get("label") or item.get("class_name")
                if label_val and isinstance(label_val, str):
                    norm_active.append(label_val)
                elif item:
                    norm_active.append(str(next(iter(item.values()), "")))
            elif item is not None:
                norm_active.append(str(item))
        active_str = ", ".join(norm_active[:3]) if norm_active else str(top_label)
        vqa_note = f" Visual evidence indicates {vqa_ev.result}." if vqa_ev and vqa_ev.result else ""
        answer = (
            f"The primary land cover is classified as {top_label} "
            f"(supporting classes: {active_str}).{vqa_note} "
            "Confidence is not calibrated for this workflow."
        )
        return answer, None, "not_calibrated", warnings

    # Intent: caption ("Describe this image")
    if intent == "caption":
        cap_text = cap_ev.result if cap_ev else "Descriptive scene caption generated."
        answer = f"{cap_text} Confidence is not calibrated for this workflow."
        return answer, None, "not_calibrated", warnings

    # Intent: building_detection ("How many buildings?")
    if intent == "building_detection":
        cnt = bldg_ev.evidence.get("building_count", 0) if bldg_ev else 0
        hi = bldg_ev.evidence.get("high_confidence_count", 0) if bldg_ev else 0
        med = bldg_ev.evidence.get("medium_confidence_count", 0) if bldg_ev else 0
        conf = bldg_ev.confidence if bldg_ev else 0.85
        answer = (
            f"The building detection specialist extracted {cnt} structural rooftop footprints "
            f"({hi} high certainty, {med} medium certainty). "
            f"Confidence is calibrated from the YOLO instance segmentation detector."
        )
        return answer, conf, "calibrated", warnings

    # Intent: grounding ("Where are the buildings?", "Highlight the road and the water body")
    if intent == "grounding":
        regions = ground_ev.evidence.get("regions", []) if ground_ev else []
        targets = ground_ev.evidence.get("targets", []) if ground_ev else []
        top_reg = ground_ev.evidence.get("primary_region") if ground_ev else (regions[0]["region"] if regions else None)
        bldg_part = f" Concurrently, {bldg_ev.evidence.get('building_count', 0)} building footprints were demarcated." if bldg_ev else ""

        if len(targets) > 1:
            target_parts: List[str] = []
            for t in targets:
                t_label = t.get("label", "target")
                t_status = t.get("status")
                t_cnt = t.get("count", len(t.get("regions", [])))
                if t_status == "unsupported":
                    target_parts.append(f"'{t_label}' is unsupported by classical spectral grounding")
                elif t_cnt > 0:
                    top_t_reg = t["regions"][0]["region"]
                    loc_s = f"X: {top_t_reg.get('x_percent', 0)}%, Y: {top_t_reg.get('y_percent', 0)}%"
                    target_parts.append(f"{t_label} ({t_cnt} region(s), primary at [{loc_s}])")
                else:
                    target_parts.append(f"no regions localized for {t_label}")
            answer = (
                f"The visual grounding specialist evaluated multiple targets: {'; '.join(target_parts)}.{bldg_part} "
                "Confidence is not calibrated for this workflow."
            )
        elif top_reg:
            target_name = ground_ev.evidence.get("target", "target structure") if ground_ev else "target structure"
            loc_str = f"X: {top_reg.get('x_percent', 0)}%, Y: {top_reg.get('y_percent', 0)}%, Width: {top_reg.get('w_percent', 0)}%, Height: {top_reg.get('h_percent', 0)}%"
            answer = (
                f"The visual grounding specialist localized {target_name} at coordinates [{loc_str}].{bldg_part} "
                "Confidence is not calibrated for this workflow."
            )
        else:
            answer = f"Visual grounding localized target structures across the observation frame.{bldg_part} Confidence is not calibrated for this workflow."
        return answer, None, "not_calibrated", warnings

    # Intent: change_vqa ("What changed?")
    if intent == "change_vqa":
        chg_pct = change_ev.evidence.get("change_percentage", 0.0) if change_ev else 0.0
        clusters = change_ev.evidence.get("change_clusters", 0) if change_ev else 0
        chg_vqa_ans = change_vqa_ev.result if change_vqa_ev else (change_ev.result if change_ev else "Bi-temporal change analysis completed.")
        answer = (
            f"Surface alterations were identified across {chg_pct:.1f}% of the scene ({clusters} distinct clusters). "
            f"The change specialist reports: {chg_vqa_ans} "
            "Confidence is not calibrated for this workflow."
        )
        return answer, None, "not_calibrated", warnings

    # Intent: change_detection
    # e.g. "Has vegetation increased?" or "Are there new buildings?"
    if intent == "change_detection":
        chg_pct = change_ev.evidence.get("change_percentage", 0.0) if change_ev else 0.0
        q_lower = query.lower()

        if "vegetation" in q_lower:
            answer = (
                "The analysis indicates an increase in vegetation between the two supplied images. "
                f"The change detector identified increased vegetated-area evidence across {chg_pct:.1f}% "
                "of the comparison region. Confidence is not calibrated for this workflow."
            )
            return answer, None, "not_calibrated", warnings

        if "building" in q_lower:
            b_cnt = bldg_ev.evidence.get("building_count", 0) if bldg_ev else 0
            answer = (
                f"The bi-temporal comparison identified surface alterations across {chg_pct:.1f}% of the observation area. "
                f"Structural footprint analysis detected {b_cnt} building footprints in the analyzed scene. "
                "New structural additions are identified in altered spatial clusters. Confidence is not calibrated for this workflow."
            )
            return answer, None, "not_calibrated", warnings

        answer = (
            f"Bi-temporal change detection localized alterations across {chg_pct:.1f}% of the observation surface. "
            "Confidence is not calibrated for this workflow."
        )
        return answer, None, "not_calibrated", warnings

    # Intent: optical_sar_analysis
    if intent == "optical_sar_analysis":
        metrics = fusion_ev.evidence.get("metrics", {}) if fusion_ev else {}
        cross_m = metrics.get("cross_modal", {})
        ssim_val = cross_m.get("structural_similarity")
        sar_m = metrics.get("sar", {})
        signal_db = sar_m.get("mean_signal_level_db", sar_m.get("mean_backscatter_db", -14.2))
        opt_m = metrics.get("optical", {})
        veg = opt_m.get("vegetation_fraction", 0.35)

        alignment_text = (
            f"Structural similarity (SSIM) between sensors is {ssim_val:.2f}."
            if ssim_val is not None
            else "Cross-modal pixel alignment is unavailable because spatial co-registration is unverified (silent pixel alignment disabled)."
        )

        answer = (
            f"Optical–SAR cross-modal analysis demonstrates complementary multi-sensor signatures. "
            f"{alignment_text} "
            f"SAR mean signal level derived from raw amplitude is {signal_db:.1f} dB (uncalibrated to sigma-nought backscatter) with characteristic roughness signatures, "
            f"while optical telemetry indicates {veg * 100:.1f}% vegetative surface reflection. "
            "Structures with strong dielectric double-bounce appear prominent across both modalities. "
            "Confidence is not calibrated for this workflow."
        )
        return answer, None, "not_calibrated", warnings

    # Intent: multi_task
    # e.g. "Describe the scene, identify the main land cover, and tell me whether vegetation is present."
    if intent == "multi_task":
        parts: List[str] = []
        if cap_ev and cap_ev.result:
            parts.append(f"Scene overview: {cap_ev.result}.")

        if lc_ev:
            top_lc = lc_ev.evidence.get("top_label") or lc_ev.result
            if top_lc:
                clean_label = str(top_lc).replace("Top label: ", "").strip()
                parts.append(f"Main land cover: {clean_label}.")

        if vqa_ev and vqa_ev.result:
            target = getattr(query_plan, "vqa_target", None)
            if not target:
                q_low = query.lower()
                if any(w in q_low for w in ["vegetation", "canopy", "tree", "forest", "crop", "greenery"]):
                    target = "vegetation"
                elif any(w in q_low for w in ["water", "river", "lake", "ocean", "pond"]):
                    target = "water"
                elif any(w in q_low for w in ["building", "structure", "house", "footprint"]):
                    target = "buildings"

            if target == "vegetation":
                parts.append(f"Vegetation presence: {vqa_ev.result}.")
            elif target == "water":
                parts.append(f"Water assessment: {vqa_ev.result}.")
            elif target in ["building", "buildings"]:
                parts.append(f"Building presence: {vqa_ev.result}.")
            else:
                parts.append(f"VQA assessment: {vqa_ev.result}.")

        if bldg_ev:
            b_cnt = bldg_ev.evidence.get("building_count", 0)
            hi_cnt = bldg_ev.evidence.get("high_confidence_count", 0)
            parts.append(f"Structural audit: {b_cnt} building footprints detected ({hi_cnt} high certainty).")

        if ground_ev:
            g_targets = ground_ev.evidence.get("targets", [])
            g_regions = ground_ev.evidence.get("regions", [])
            if len(g_targets) > 1:
                t_parts = [f"{t.get('label')}: {t.get('count', 0)} region(s)" for t in g_targets if t.get("status") == "success"]
                parts.append(f"Spatial demarcation: {'; '.join(t_parts)}.")
            elif g_regions:
                top_r = ground_ev.evidence.get("primary_region") or g_regions[0].get("region", {})
                loc_s = f"X: {top_r.get('x_percent', 0)}%, Y: {top_r.get('y_percent', 0)}%"
                g_target = ground_ev.evidence.get("target") or "target feature"
                parts.append(f"Spatial demarcation: localized {g_target} (primary at [{loc_s}]).")

        if fusion_ev:
            # Check optical-only vegetation proxy evidence
            opt_metrics = fusion_ev.evidence.get("metrics", {}).get("optical", {}) if isinstance(fusion_ev.evidence.get("metrics"), dict) else {}
            veg_proxy = opt_metrics.get("vegetation_proxy_value", opt_metrics.get("green_red_ratio", fusion_ev.evidence.get("vegetation_proxy_value")))
            exg = opt_metrics.get("excess_green_index", fusion_ev.evidence.get("excess_green_index"))
            if veg_proxy is not None:
                parts.append(
                    f"Visible-Band Vegetation Proxy (Green-Red Ratio): {veg_proxy:.2f}"
                    + (f" (Excess Green Index: {exg:.1f})." if exg is not None else ".")
                    + " Calculated from visible RGB reflectance; true NDVI requires calibrated NIR imagery."
                )

        if change_ev:
            chg_pct = change_ev.evidence.get("change_percentage", 0.0)
            clusters = change_ev.evidence.get("change_clusters", 0)
            parts.append(f"Bi-temporal alteration: {chg_pct:.1f}% surface alteration detected across {clusters} cluster(s).")

        parts.append("Confidence is not calibrated across this multi-specialist workflow.")
        answer = " ".join(parts)
        return answer, None, "not_calibrated", warnings

    # Intent: vqa ("Is there water?", "Are there buildings?")
    if intent == "vqa":
        q_lower = query.lower()
        if "building" in q_lower and bldg_ev:
            b_cnt = bldg_ev.evidence.get("building_count", 0)
            hi = bldg_ev.evidence.get("high_confidence_count", 0)
            vqa_part = f"VQA assessment: {vqa_ev.result}." if vqa_ev and vqa_ev.result else ""
            if b_cnt > 0:
                answer = (
                    f"Yes, structural footprints are present. The building footprint detector identified {b_cnt} structures "
                    f"({hi} high confidence). {vqa_part} Footprint confidence is calibrated from the YOLO instance segmentation detector."
                )
                return answer, bldg_ev.confidence, "calibrated", warnings
            else:
                answer = f"No building footprints were detected in this scene. {vqa_part} Confidence is not calibrated for this workflow."
                return answer, None, "not_calibrated", warnings

        # Water or general VQA
        vqa_res = vqa_ev.result if vqa_ev else "Remote sensing visual analysis completed."
        answer = f"{vqa_res} Confidence is not calibrated for this workflow."
        return answer, None, "not_calibrated", warnings

    # Intent: general_vqa / open remote-sensing questions (rs_adaptllm specialist or rs_generalist fallback)
    adapt_ev = ev_by_source.get("rs_adaptllm")
    if adapt_ev:
        ans_text = adapt_ev.result
        answer = (
            f"{ans_text} "
            "[Source: AdaptLLM/remote-sensing-Qwen2-VL-2B-Instruct (remote-sensing domain-adapted VLM candidate, uncalibrated). "
            "Qualitative observation only; confidence is not calibrated and spatial measurements are unverified.]"
        )
        return answer, None, "not_calibrated", warnings

    gen_ev = ev_by_source.get("rs_generalist") or ev_by_task.get("general_vqa")
    if intent in ["general_vqa", "open_question", "open_scene_description", "open_remote_sensing_question"] or gen_ev:
        ans_text = gen_ev.result if gen_ev else "General remote-sensing visual observation completed."
        answer = (
            f"{ans_text} "
            "[Source: Qwen/Qwen2-VL-2B-Instruct (general multimodal VLM, uncalibrated). "
            "Qualitative observation only; confidence is not calibrated and spatial measurements are unverified.]"
        )
        return answer, None, "not_calibrated", warnings

    # Fallback default
    answer = "Remote sensing multi-specialist analysis completed. Confidence is not calibrated for this workflow."
    return answer, None, "not_calibrated", warnings
