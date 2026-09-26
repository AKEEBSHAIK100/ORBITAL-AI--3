"""Query planner regression matrix for judge-style remote-sensing prompts.

These tests validate routing only. They do not claim that a specialist's model weights
or external services are available; runtime availability is tested separately.
"""
import pytest

from backend.agents.planner import create_query_plan


@pytest.mark.parametrize(
    "query,images,expected_intent,expected_specialist",
    [
        ("Describe this satellite image", 1, "caption", "rs_caption_adapted"),
        ("What can you tell me about this scene?", 1, "caption", "rs_caption_adapted"),
        ("What features are visible in this image?", 1, "vqa", "rs_vqa_adapted"),
        ("Is there water in this image?", 1, "vqa", "rs_vqa_adapted"),
        ("Are there buildings in this image?", 1, "vqa", "rs_vqa_adapted"),
        ("What type of land cover is shown?", 1, "land_cover", "land_cover"),
        ("Is this area urban?", 1, "vqa", "rs_vqa_adapted"),
        ("Locate the roads in the image", 1, "grounding", "visual_grounding"),
        ("Where are the buildings?", 1, "grounding", "visual_grounding"),
        ("How many buildings are visible?", 1, "building_detection", "building_detection"),
        ("Are there ships in the harbor?", 1, "general_vqa", None),
        ("Where are the solar panels?", 1, "general_vqa", None),
        ("Compare these images and tell me what changed", 2, "change_vqa", "change_vqa"),
        ("Did vegetation increase between these images?", 2, "change_detection", "change_detection"),
        ("Were new buildings added?", 2, "change_detection", "change_detection"),
        ("Analyze the optical and SAR images together", 2, "optical_sar_analysis", "optical_sar_fusion"),
        ("Describe the scene and identify the main land cover", 1, "multi_task", "rs_caption_adapted"),
        ("What can you tell me about this image?", 1, "caption", "rs_caption_adapted"),
        ("Hello, what can you do?", 1, "unsupported", None),
    ],
)
def test_judge_query_routing(query, images, expected_intent, expected_specialist):
    plan = create_query_plan(query, image_count=images, modalities=["optical"])
    assert plan.intent == expected_intent
    if expected_specialist:
        assert expected_specialist in plan.specialists


@pytest.mark.parametrize(
    "query",
    [
        "How many people live here?",
        "Who is this person?",
        "What is the weather tomorrow?",
        "What is the stock price?",
        "Give me a medical diagnosis",
        "What is the president's approval rating?",
        "Write me Python code",
    ],
)
def test_non_rs_queries_are_not_routed_to_visual_specialists(query):
    plan = create_query_plan(query, image_count=1, modalities=["optical"])
    assert plan.intent == "unsupported"
    assert plan.planner_disposition == "No supported visual capability"
