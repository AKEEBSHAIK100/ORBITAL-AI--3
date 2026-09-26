from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_free_only_api_surface_has_no_paid_provider_client():
    targets = [
        ROOT / "api" / "_lib.ts",
        ROOT / "api" / "analyze.ts",
        ROOT / "api" / "compare.ts",
        ROOT / "api" / "fuse.ts",
        ROOT / "server.ts",
    ]
    forbidden = ("from 'openai'", 'from "openai"', "new OpenAI(", "client.chat.completions")
    for path in targets:
        text = path.read_text(encoding="utf-8")
        assert not any(token in text for token in forbidden), path


def test_no_known_synthetic_comparison_fallback_remains():
    targets = [
        ROOT / "api" / "_lib.ts",
        ROOT / "api" / "analyze.ts",
        ROOT / "api" / "compare.ts",
        ROOT / "api" / "fuse.ts",
        ROOT / "server.ts",
    ]
    forbidden = (
        "12.4%",
        "generateRealisticComparison",
        "calibrated simulation",
        "Descriptive scene caption generated.",
    )
    for path in targets:
        text = path.read_text(encoding="utf-8")
        assert not any(token in text for token in forbidden), path


def test_client_and_server_runtime_model_labels_match():
    client = (ROOT / "src" / "lib" / "constants.ts").read_text(encoding="utf-8")
    server = (ROOT / "lib" / "constants.ts").read_text(encoding="utf-8")

    def model_value(text):
        marker = "export const MODEL = "
        line = next(line for line in text.splitlines() if marker in line)
        return line.split(marker, 1)[1].strip()

    assert model_value(client) == model_value(server)


def test_blip_adapter_configs_are_qkv_only_and_match_base_models():
    import json

    expected = {
        "blip_rs_lora": "Salesforce/blip-image-captioning-base",
        "blip_vqa_rs_lora": "Salesforce/blip-vqa-base",
    }
    for name, base in expected.items():
        path = ROOT / "backend" / "models" / "adapters" / name / "adapter_config.json"
        cfg = json.loads(path.read_text(encoding="utf-8"))
        assert cfg["base_model_name_or_path"] == base
        assert cfg["target_modules"] == ["qkv"]
        assert cfg["task_type"] == "FEATURE_EXTRACTION"
        assert "dense" not in cfg["target_modules"]


def test_blip_weight_absence_is_not_hidden_by_placeholder_files():
    for name in ("blip_rs_lora", "blip_vqa_rs_lora"):
        weights = ROOT / "backend" / "models" / "adapters" / name / "adapter_model.safetensors"
        if weights.exists():
            assert weights.stat().st_size > 0
