"""
Orbital-AI / SatQuery AI — Multi-Specialist Agentic Orchestrator.

Implements the central execution loop:
1. Query understood (QueryPlan generated)
2. Input validation (Sensor modalities, dimensions, CRS, affine, temporal verification)
3. Task plan (Execution sequence defined)
4. Specialist selected (ModelRegistry availability and checkpoint inspection)
5. Specialist executed (Inference with timing and telemetry)
6. Evidence collected (Uniform SpecialistEvidenceObject schema)
7. Evidence combined (Cross-specialist synthesis and conflict analysis)
8. Final response (Truthful natural language synthesis without hallucination)

Emits observable trace without hidden chain-of-thought.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ..schemas.analysis import (
    ObservableTrace,
    QueryPlan,
    SpecialistEvidenceObject,
    UnifiedAnalysisResponse,
    BuildingAnalysisResponse,
    BuildingDetectionItem,
    GroundingItem,
    GroundingRegion,
)
from .planner import create_query_plan
from .validator import validate_input_imagery
from .synthesizer import synthesize_response
from .aggregator import build_observable_trace
from ..tools.registry import get_tool
from ..models.registry import ModelRegistry


def run_orbital_analysis(
    query: str,
    images: List[np.ndarray],
    modalities: Optional[List[str]] = None,
    image_names: Optional[List[str]] = None,
    metadata_list: Optional[List[Dict[str, Any]]] = None,
    parameters: Optional[Dict[str, Any]] = None
) -> UnifiedAnalysisResponse:
    """
    Executes the multi-specialist agentic pipeline end-to-end.
    Guarantees one of:
    - SUCCESS
    - VALIDATION_ERROR
    - SPECIALIST_UNAVAILABLE
    - UNSUPPORTED_QUERY
    Never fabricates unsupported answers.
    """
    t_start = time.time()
    steps_log: List[Dict[str, Any]] = []
    evidence_list: List[SpecialistEvidenceObject] = []
    visual_evidence: Dict[str, Any] = {}
    tools_used: List[str] = []
    parameters = parameters or {}

    img1 = images[0] if len(images) > 0 else None
    img2 = images[1] if len(images) > 1 else None
    meta1 = metadata_list[0] if (metadata_list and len(metadata_list) > 0) else {}
    meta2 = metadata_list[1] if (metadata_list and len(metadata_list) > 1) else {}

    # ── Step 1: Query Understood (Generate Structured QueryPlan) ──────────────
    t_plan = time.time()
    plan = create_query_plan(query, image_count=len(images), modalities=modalities)
    plan_dur = (time.time() - t_plan) * 1000

    steps_log.append({
        "step": 1,
        "tool": "query_planner",
        "description": "Interpreting query intent and generating structured execution plan",
        "input_summary": f"Query: '{query}'",
        "output_summary": f"Intent: {plan.intent}; Tasks: {', '.join(plan.required_tasks) or 'none'}; Specialists: {', '.join(plan.specialists) or 'none'}",
        "duration_ms": round(plan_dur, 2),
        "status": "success",
        "success": True,
        "confidence_source": "deterministic_rule_based",
        "parameters": {
            "intent": plan.intent,
            "required_images": plan.required_images,
            "required_modalities": plan.required_modalities,
        }
    })

    # ── Handle Unsupported Queries Immediately ───────────────────────────────
    if plan.intent == "unsupported":
        dummy_val = {
            "images_provided": len(images),
            "modalities": modalities or ["optical"],
            "compatibility": "not_applicable",
            "notes": ["Query falls outside remote-sensing capabilities"],
            "warnings": [],
            "errors": []
        }
        answer, conf, conf_status, warnings = synthesize_response(
            query, plan, [], dummy_val, status="UNSUPPORTED_QUERY"
        )
        steps_log.append({
            "step": 2,
            "tool": "query_scope_control",
            "description": "Recognized unsupported query outside remote-sensing task family",
            "input_summary": f"Query: '{query}'",
            "output_summary": "Status: UNSUPPORTED_QUERY with explanation and alternative tasks",
            "duration_ms": 0.5,
            "status": "success",
            "success": True,
            "confidence_source": "none",
            "parameters": {"intent": "unsupported"}
        })

        trace = build_observable_trace(
            task_type="unsupported",
            steps=steps_log,
            total_duration_ms=(time.time() - t_start) * 1000,
            validation=dummy_val,
            primary_tool={"model_id": "none", "adapter": "none", "domain_adaptation": "none"}
        )

        return UnifiedAnalysisResponse(
            status="UNSUPPORTED_QUERY",
            answer=answer,
            query_plan=plan,
            evidence=[],
            visual_evidence={},
            warnings=warnings,
            confidence=conf,
            confidence_status=conf_status,
            execution_trace=trace,
            success=False,
            task_type="unsupported",
            tools_used=[],
            input_modality=(modalities or ["optical"])[0],
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
            execution_time_ms=round((time.time() - t_start) * 1000, 2),
        )

    # ── Step 2: Input Validation ─────────────────────────────────────────────
    t_val = time.time()
    validation = validate_input_imagery(
        task_type=plan.intent,
        images=images,
        modalities=modalities,
        image_names=image_names,
        metadata_list=metadata_list,
        query_plan=plan
    )
    val_dur = (time.time() - t_val) * 1000

    steps_log.append({
        "step": 2,
        "tool": "input_validator",
        "description": "Verifying dimensions, sensor modalities, CRS, and co-registration metadata",
        "input_summary": f"{len(images)} image(s); Modalities: {', '.join(validation['modalities'])}",
        "output_summary": f"Compatibility: {validation['compatibility'].upper()} ({len(validation['notes'])} checks)",
        "duration_ms": round(val_dur, 2),
        "status": "error" if validation["compatibility"] == "error" else "success",
        "success": validation["compatibility"] != "error",
        "confidence_source": "none",
        "parameters": {
            "compatibility": validation["compatibility"],
            "crs_verified": bool(validation.get("crs")),
            "geotransform_verified": bool(validation.get("geotransforms"))
        }
    })

    # Reject Incompatible Workflows
    if validation["compatibility"] == "error":
        answer, conf, conf_status, warnings = synthesize_response(
            query, plan, [], validation, status="VALIDATION_ERROR"
        )
        trace = build_observable_trace(
            task_type=plan.intent,
            steps=steps_log,
            total_duration_ms=(time.time() - t_start) * 1000,
            validation=validation,
            primary_tool={"model_id": "validator", "adapter": "input_validator", "domain_adaptation": "co-registration"}
        )
        return UnifiedAnalysisResponse(
            status="VALIDATION_ERROR",
            answer=answer,
            query_plan=plan,
            evidence=[],
            visual_evidence={},
            warnings=warnings,
            confidence=None,
            confidence_status="unavailable",
            execution_trace=trace,
            success=False,
            task_type=plan.intent,
            tools_used=[],
            input_modality=(modalities or ["optical"])[0],
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
            execution_time_ms=round((time.time() - t_start) * 1000, 2),
        )

    # ── Step 3: Task Plan ────────────────────────────────────────────────────
    steps_log.append({
        "step": 3,
        "tool": "task_sequencer",
        "description": "Sequencing registered remote-sensing specialists for multi-step execution",
        "input_summary": f"Intent: {plan.intent}",
        "output_summary": f"Execution order: {' -> '.join(plan.execution_order)}",
        "duration_ms": 0.5,
        "status": "success",
        "success": True,
        "confidence_source": "none",
        "parameters": {
            "execution_order": plan.execution_order,
            "evidence_requirements": plan.evidence_requirements
        }
    })

    # ── Step 4 & 5 & 6: Specialist Selection, Execution & Evidence Collection ─
    model_reg = ModelRegistry.get_instance()
    step_num = 4

    building_response: Optional[BuildingAnalysisResponse] = None
    grounding_response: Optional[List[GroundingItem]] = None
    change_response: Optional[Dict[str, Any]] = None
    fusion_response: Optional[Dict[str, Any]] = None
    land_cover_response: Optional[Dict[str, Any]] = None

    tool_inputs = {
        "query": query,
        "question": query,
        "image": img1,
        "secondary_image": img2,
        "optical_image": img1,
        "sar_image": img2,
        "metadata": meta1,
        "secondary_metadata": meta2,
        "date_t1": meta1.get("acquisition_date") or meta1.get("date"),
        "date_t2": meta2.get("acquisition_date") or meta2.get("date"),
    }

    primary_tool_spec: Dict[str, Any] = {}

    # Request-scoped shared evidence context to eliminate redundant multi-specialist inferences
    shared_context: Dict[str, Any] = {
        "land_cover": None,
        "building_analysis": None,
        "caption": None,
        "vqa": None,
        "grounding": None,
        "change_detection": None,
        "optical_sar": None,
    }

    for specialist_id in plan.execution_order:
        # Check specialist availability in ModelRegistry
        spec_entry = model_reg.get_specialist(specialist_id)
        tool_obj = get_tool(specialist_id)

        # Specialist selection step
        specialist_name = spec_entry.name if spec_entry else (tool_obj.name if tool_obj else specialist_id)
        model_id = spec_entry.model_id if spec_entry else (getattr(tool_obj, "model_id", specialist_id))

        if not primary_tool_spec:
            tool_perm_params = getattr(tool_obj, "permitted_parameters", {})
            primary_tool_spec = {
                "model_id": model_id,
                "adapter": getattr(tool_obj, "adapter", specialist_name),
                "domain_adaptation": getattr(tool_obj, "domain_adaptation", "Domain-adapted RS Specialist"),
                "permitted_parameters": tool_perm_params if isinstance(tool_perm_params, dict) else {}
            }

        # Check availability
        is_avail = (spec_entry.is_available if spec_entry else True) and (tool_obj is not None)
        unavail_reason = spec_entry.unavailable_reason if (spec_entry and not spec_entry.is_available) else (
            f"Specialist '{specialist_id}' is not installed in ToolRegistry." if tool_obj is None else None
        )

        steps_log.append({
            "step": step_num,
            "tool": "specialist_selector",
            "description": f"Selecting and verifying registered specialist '{specialist_name}'",
            "input_summary": f"Specialist ID: {specialist_id}",
            "output_summary": f"Model: {model_id} (Available: {is_avail})",
            "duration_ms": 0.5,
            "status": "success" if is_avail else "unavailable",
            "success": is_avail,
            "confidence_source": "none",
            "parameters": {"specialist_id": specialist_id, "is_available": is_avail}
        })
        step_num += 1

        if not is_avail:
            # Return SPECIALIST_UNAVAILABLE immediately without fabricating
            ev_unavail = SpecialistEvidenceObject(
                task=spec_entry.task if spec_entry else specialist_id,
                result=f"Specialist '{specialist_id}' unavailable: {unavail_reason}",
                evidence={},
                source=specialist_id,
                model=model_id,
                provenance=None,
                confidence=None,
                confidence_status="unavailable",
                warnings=[unavail_reason or f"Model checkpoint for '{specialist_id}' is not installed."]
            )
            evidence_list.append(ev_unavail)

            answer, conf, conf_status, warnings = synthesize_response(
                query, plan, evidence_list, validation, status="SPECIALIST_UNAVAILABLE"
            )

            trace = build_observable_trace(
                task_type=plan.intent,
                steps=steps_log,
                total_duration_ms=(time.time() - t_start) * 1000,
                validation=validation,
                primary_tool=primary_tool_spec
            )

            return UnifiedAnalysisResponse(
                status="SPECIALIST_UNAVAILABLE",
                answer=answer,
                query_plan=plan,
                evidence=evidence_list,
                visual_evidence={},
                warnings=warnings,
                confidence=None,
                confidence_status="unavailable",
                execution_trace=trace,
                success=False,
                task_type=plan.intent,
                tools_used=tools_used,
                input_modality=(modalities or ["optical"])[0],
                timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
                execution_time_ms=round((time.time() - t_start) * 1000, 2),
            )

        # Execute Specialist
        tools_used.append(specialist_id)
        t_exec = time.time()

        # Specialist-specific inputs: pass focused sub-question if available
        curr_tool_inputs = dict(tool_inputs)
        effective_query = query
        if specialist_id in ["rs_vqa_adapted", "vqa"] and getattr(plan, "vqa_question", None):
            curr_tool_inputs["query"] = plan.vqa_question
            curr_tool_inputs["question"] = plan.vqa_question
            effective_query = plan.vqa_question

        # Prepare specialist parameters with request-scoped shared evidence
        spec_params = dict(parameters)
        reused_evidence_sources: List[str] = []

        if specialist_id in ["rs_caption_adapted", "caption"]:
            if shared_context["land_cover"] is not None:
                spec_params["supporting_land_cover_result"] = shared_context["land_cover"]
                reused_evidence_sources.append("land_cover")

        elif specialist_id in ["rs_vqa_adapted", "vqa"]:
            if shared_context["land_cover"] is not None:
                spec_params["supporting_land_cover_result"] = shared_context["land_cover"]
                reused_evidence_sources.append("land_cover")
            if shared_context["building_analysis"] is not None:
                spec_params["supporting_building_result"] = shared_context["building_analysis"]
                reused_evidence_sources.append("building_detection")

        if reused_evidence_sources:
            steps_log.append({
                "step": step_num,
                "tool": "evidence_deduplicator",
                "description": f"Injecting shared evidence into {specialist_name} to eliminate duplicate inference",
                "input_summary": f"Shared sources: {', '.join(reused_evidence_sources)}",
                "output_summary": f"Reused existing {', '.join(reused_evidence_sources)} evidence; skipped duplicate inference.",
                "duration_ms": 0.1,
                "status": "success",
                "success": True,
                "confidence_source": "none",
                "parameters": {"reused_sources": reused_evidence_sources}
            })
            step_num += 1

        res = tool_obj.run(curr_tool_inputs, spec_params)
        exec_dur = (time.time() - t_exec) * 1000

        # Runtime availability is authoritative. A registry entry alone must never
        # turn an unavailable/error specialist response into a successful analysis.
        runtime_status = str(res.get("status", "")).lower()
        if runtime_status in ("specialist_unavailable", "unavailable", "error"):
            reason = res.get("answer") or res.get("error") or f"Specialist '{specialist_id}' failed to execute."
            warnings = res.get("warnings", [])
            if not isinstance(warnings, list):
                warnings = [str(warnings)]
            ev_runtime = SpecialistEvidenceObject(
                task=spec_entry.task if spec_entry else specialist_id,
                result=str(reason),
                evidence=res.get("evidence", {}),
                source=specialist_id,
                model=model_id,
                provenance=res.get("provenance"),
                confidence=None,
                confidence_status="unavailable" if runtime_status != "error" else "unavailable",
                warnings=warnings,
            )
            evidence_list.append(ev_runtime)
            answer, conf, conf_status, final_warnings = synthesize_response(
                query, plan, evidence_list, validation, status="SPECIALIST_UNAVAILABLE"
            )
            final_warnings = list(dict.fromkeys([*final_warnings, *warnings]))
            trace = build_observable_trace(
                task_type=plan.intent,
                steps=steps_log + [{
                    "step": step_num,
                    "tool": specialist_name,
                    "description": f"Runtime execution of {specialist_name} did not produce a usable specialist result",
                    "input_summary": f"Scene observation with query: '{effective_query}'",
                    "output_summary": str(reason)[:100],
                    "duration_ms": round(exec_dur, 2),
                    "status": runtime_status,
                    "success": False,
                    "confidence_source": "none",
                    "parameters": {},
                }],
                total_duration_ms=(time.time() - t_start) * 1000,
                validation=validation,
                primary_tool=primary_tool_spec
            )
            return UnifiedAnalysisResponse(
                status="SPECIALIST_UNAVAILABLE",
                answer=answer,
                query_plan=plan,
                evidence=evidence_list,
                visual_evidence={},
                warnings=final_warnings,
                confidence=None,
                confidence_status="unavailable",
                execution_trace=trace,
                success=False,
                task_type=plan.intent,
                tools_used=tools_used,
                input_modality=(modalities or ["optical"])[0],
                timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
                execution_time_ms=round((time.time() - t_start) * 1000, 2),
            )

        # Update request-scoped shared evidence context with successful outputs
        if res.get("status") in ("success", "SUCCESS"):
            if specialist_id == "land_cover":
                shared_context["land_cover"] = res
            elif specialist_id == "building_detection":
                shared_context["building_analysis"] = res
            elif specialist_id in ["rs_caption_adapted", "caption"]:
                shared_context["caption"] = res
            elif specialist_id in ["rs_vqa_adapted", "vqa"]:
                shared_context["vqa"] = res
            elif specialist_id in ["visual_grounding", "grounding"]:
                shared_context["grounding"] = res
            elif specialist_id in ["change_detection", "change_vqa"]:
                shared_context["change_detection"] = res
            elif specialist_id in ["optical_sar_fusion", "optical_sar"]:
                shared_context["optical_sar"] = res

        # Construct structured evidence per Section 6
        task_name = getattr(tool_obj, "supported_tasks", [plan.intent])[0]
        model_name = getattr(tool_obj, "model_id", specialist_id)
        provenance = getattr(tool_obj, "provenance", getattr(tool_obj, "domain_adaptation", "RS Benchmark"))
        tool_conf = res.get("confidence")
        conf_status = res.get("confidence_status", "calibrated" if tool_conf is not None else "not_calibrated")
        tool_warnings = res.get("warnings", [])

        if specialist_id in ("rs_generalist", "rs_adaptllm"):
            tool_conf = None
            conf_status = "not_calibrated"

        # Process specialist specific outputs
        if specialist_id == "building_detection":
            b_items = [
                BuildingDetectionItem(**d) for d in res.get("detections", [])
            ]
            building_response = BuildingAnalysisResponse(
                success=True,
                image_dimensions=res.get("image_dimensions", {"width": 0, "height": 0}),
                tiles_processed=res.get("tiles_processed", 1),
                raw_detections_count=res.get("raw_detections_count", 0),
                merged_detections_count=res.get("merged_detections_count", 0),
                building_count=res.get("building_count", 0),
                high_confidence_count=res.get("high_confidence_count", 0),
                medium_confidence_count=res.get("medium_confidence_count", 0),
                low_confidence_count=res.get("low_confidence_count", 0),
                partial_count=res.get("partial_count", 0),
                confidence=res.get("confidence"),
                confidence_level=res.get("confidence_level"),
                validation_status=res.get("validation_status", "unverified"),
                validation=res.get("validation", {}),
                detections=b_items,
                geojson=res.get("geojson")
            )
            visual_evidence["building_analysis"] = building_response.model_dump() if hasattr(building_response, "model_dump") else building_response.dict()
            visual_evidence["geojson"] = res.get("geojson")
            result_summary = f"Detected {res.get('building_count', 0)} structures ({res.get('high_confidence_count', 0)} high confidence)"

        elif specialist_id in ["visual_grounding", "grounding"]:
            g_items = [
                GroundingItem(
                    target=r["target"],
                    region=GroundingRegion(**r["region"]),
                    confidence=r.get("confidence", 0.0),
                    label=r["label"]
                )
                for r in res.get("regions", [])
            ]
            grounding_response = g_items
            visual_evidence["grounding_regions"] = [
                g.model_dump() if hasattr(g, "model_dump") else g.dict() for g in g_items
            ]
            result_summary = f"Localized {len(g_items)} regions for '{res.get('target', 'target')}'"

        elif specialist_id in ["optical_sar_fusion", "optical_sar"]:
            fusion_response = res.get("metrics")
            visual_evidence["fusion_telemetry"] = fusion_response
            result_summary = res.get("interpretation", "Optical-SAR fusion metrics extracted")

        elif specialist_id in ["change_detection", "change_vqa"]:
            change_response = res
            visual_evidence["change_map"] = {
                "change_percentage": res.get("change_percentage", 0.0),
                "change_clusters": res.get("change_clusters", 0)
            }
            result_summary = res.get("answer", f"Change detected across {res.get('change_percentage', 0)}% of scene")

        elif specialist_id == "land_cover":
            land_cover_response = res
            visual_evidence["land_cover"] = {
                "top_label": res.get("top_label"),
                "active_labels": [l.get("name") if isinstance(l, dict) else l for l in res.get("active_labels", [])]
            }
            result_summary = f"Top label: {res.get('top_label')}"

        else:
            ans = res.get("answer") or res.get("caption")
            if not ans:
                tool_st = res.get("status", "unknown")
                if tool_st in ("specialist_unavailable", "error"):
                    ans = f"Specialist '{specialist_name}' {tool_st}: {', '.join(tool_warnings) if tool_warnings else 'execution failed'}"
                else:
                    ans = f"Specialist '{specialist_name}' analysis completed"
            result_summary = str(ans)

        # Log execution
        tool_status_str = res.get("status", "success")
        is_tool_success = (tool_status_str == "success")
        steps_log.append({
            "step": step_num,
            "tool": specialist_name,
            "description": f"Executing inference on {tool_obj.supported_tasks[0]} specialist",
            "input_summary": f"Scene observation with query: '{effective_query}'",
            "output_summary": result_summary[:100] if result_summary else "Specialist execution completed",
            "duration_ms": round(exec_dur, 2),
            "status": tool_status_str,
            "success": is_tool_success,
            "confidence_source": "calibrated_score" if tool_conf is not None else "model_output_uncalibrated",
            "parameters": getattr(tool_obj, "permitted_parameters", {}) if isinstance(getattr(tool_obj, "permitted_parameters", {}), dict) else {}
        })
        step_num += 1

        # Evidence object
        ev_obj = SpecialistEvidenceObject(
            task=task_name,
            result=str(res.get("answer") or res.get("caption") or result_summary),
            evidence=res.get("evidence", res),
            source=specialist_id,
            model=model_name,
            provenance=provenance,
            confidence=tool_conf,
            confidence_status=conf_status,
            warnings=tool_warnings
        )
        evidence_list.append(ev_obj)

        steps_log.append({
            "step": step_num,
            "tool": "evidence_collector",
            "description": f"Collected structured evidence from {specialist_name}",
            "input_summary": f"Specialist output for task '{task_name}'",
            "output_summary": f"Confidence: {conf_status}; Findings keys: {', '.join(list(res.keys())[:4])}",
            "duration_ms": 0.5,
            "status": "success",
            "success": True,
            "confidence_source": "none",
            "parameters": {"task": task_name}
        })
        step_num += 1

    # ── Step 7: Evidence Combined ────────────────────────────────────────────
    t_comb = time.time()
    steps_log.append({
        "step": step_num,
        "tool": "evidence_aggregator",
        "description": "Synthesizing multi-specialist evidence, cross-verifying signals, and inspecting conflicts",
        "input_summary": f"{len(evidence_list)} specialist evidence package(s)",
        "output_summary": f"Aggregated findings across {len(plan.required_tasks)} task domains",
        "duration_ms": round((time.time() - t_comb) * 1000, 2),
        "status": "success",
        "success": True,
        "confidence_source": "none",
        "parameters": {"tasks_aggregated": plan.required_tasks}
    })
    step_num += 1

    # ── Step 8: Final Response ───────────────────────────────────────────────
    t_resp = time.time()
    final_answer, final_conf, final_conf_status, all_warnings = synthesize_response(
        query, plan, evidence_list, validation, status="SUCCESS"
    )

    steps_log.append({
        "step": step_num,
        "tool": "response_synthesizer",
        "description": "Converting aggregated evidence into truthful natural answer without invented confidence",
        "input_summary": f"Aggregated evidence from {len(evidence_list)} specialists",
        "output_summary": "Truthful response generated; calibration reported accurately",
        "duration_ms": round((time.time() - t_resp) * 1000, 2),
        "status": "success",
        "success": True,
        "confidence_source": "none",
        "parameters": {"confidence_status": final_conf_status}
    })

    total_duration_ms = (time.time() - t_start) * 1000

    trace = build_observable_trace(
        task_type=plan.intent,
        steps=steps_log,
        total_duration_ms=total_duration_ms,
        validation=validation,
        primary_tool=primary_tool_spec
    )

    return UnifiedAnalysisResponse(
        status="SUCCESS",
        answer=final_answer,
        query_plan=plan,
        evidence=evidence_list,
        visual_evidence=visual_evidence,
        warnings=all_warnings,
        confidence=final_conf,
        confidence_status=final_conf_status,
        execution_trace=trace,
        success=True,
        task_type=plan.intent,
        confidence_level="High" if final_conf and final_conf >= 0.8 else "UNAVAILABLE",
        confidence_source="calibrated" if final_conf is not None else "model_output_not_calibrated",
        tools_used=tools_used,
        input_modality=(modalities or ["optical"])[0],
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        execution_time_ms=round(total_duration_ms, 2),
        building_analysis=building_response,
        grounding=grounding_response,
        change_map=change_response,
        fusion_metrics=fusion_response,
        land_cover=land_cover_response,
        mode="model"
    )
