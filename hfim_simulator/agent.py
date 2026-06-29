from __future__ import annotations

import os
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "gemini-2.5-flash"


def load_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def build_agent_context(
    system: dict[str, Any],
    setup_drug_name: str,
    drug_inputs: dict[str, dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "system": system,
        "setup_drug_name": setup_drug_name,
        "drug_inputs": drug_inputs,
        "summary": summary,
    }


def ask_setup_agent(
    question: str,
    context: dict[str, Any],
    *,
    env: dict[str, str] | None = None,
    project_root: Path | None = None,
    use_gemini: bool = True,
) -> dict[str, str]:
    merged_env = dict(os.environ)
    if project_root is not None:
        merged_env.update(load_env_file(project_root / ".env"))
    if env is not None:
        merged_env.update(env)

    api_key = merged_env.get("GEMINI_API_KEY", "")
    if use_gemini and api_key:
        try:
            message = _ask_gemini(question, context, api_key, merged_env.get("GEMINI_MODEL", DEFAULT_MODEL))
            return {"source": "gemini", "message": message}
        except Exception as exc:  # pragma: no cover - exercised manually with real API credentials.
            fallback = rule_based_setup_reply(question, context)
            return {
                "source": "local_rules",
                "message": f"Gemini is not available right now, so I used local HFIM rules instead.\n\n{fallback}\n\nTechnical note: {exc.__class__.__name__}",
            }

    return {"source": "local_rules", "message": rule_based_setup_reply(question, context)}


def rule_based_setup_reply(question: str, context: dict[str, Any]) -> str:
    system = context.get("system", {})
    drug_inputs = context.get("drug_inputs", {})
    summary = context.get("summary", {})
    setup_drug_name = context.get("setup_drug_name", "selected drug")

    lines = [
        "I checked the setup with local HFIM/PK rules, so no external model is required for this response.",
        "",
        f"- Central effective volume: {system.get('central_volume_ml', 'unknown')} mL",
        f"- Extra volume: {system.get('extra_volume_ml', 'unknown')} mL",
        f"- Central/extra setup drug: {setup_drug_name}",
    ]

    for name, values in drug_inputs.items():
        item = summary.get(name, {})
        lines.append("")
        lines.append(f"**{name}**")
        lines.append(f"- Target: {values.get('target_concentration_mg_l', values.get('target_value', 'unknown'))} mg/L")
        lines.append(f"- Half-life: {values.get('half_life_h', 'unknown')} h")
        if values.get("loading_dose"):
            loading_target = values.get("loading_target_concentration_mg_l")
            loading_duration = values.get("loading_duration_h", 0)
            loading_amount = item.get("loading_dose_mg")
            lines.append(f"- Loading target: {loading_target} mg/L over {loading_duration:g} h")
            if loading_amount is not None:
                lines.append(f"- Estimated loading amount: {loading_amount:.3f} mg")
            if loading_duration and loading_duration > 0:
                lines.append("- Because loading is infused over time, the central concentration should rise gradually instead of jumping at t=0.")
        else:
            lines.append("- Loading dose: not selected")

        maintenance = values.get("maintenance", "unknown")
        lines.append(f"- Maintenance: {maintenance}")
        if maintenance == "continuous infusion" and "infusion_rate_mg_h" in item:
            lines.append(f"- Continuous infusion rate: {item['infusion_rate_mg_h']:.3f} mg/h")
        if maintenance == "intermittent infusion":
            lines.append(f"- Dosing frequency: q{values.get('dosing_frequency_h', 'unknown')}h")
            lines.append(f"- Infusion duration: {values.get('maintenance_duration_h', 'unknown')} h")

    lines.extend([
        "",
        "Suggested next step: if you want the assistant to fill values automatically, have it return structured JSON first, then validate the values with the local PK calculator before updating the simulator state.",
    ])
    return "\n".join(lines)


def _ask_gemini(question: str, context: dict[str, Any], api_key: str, model: str) -> str:
    from google import genai

    client = genai.Client(api_key=api_key)
    prompt = _build_prompt(question, context)
    response = client.models.generate_content(model=model, contents=prompt)
    return getattr(response, "text", str(response))


def _build_prompt(question: str, context: dict[str, Any]) -> str:
    return (
        "You are an HFIM pharmacometrics setup assistant. "
        "Use deterministic PK reasoning, state assumptions, and never invent missing experimental constraints. "
        "Do not ask for or reveal API keys. Reply in clear scientific English unless the user asks for another language.\n\n"
        f"Current simulator context:\n{context}\n\n"
        f"User question:\n{question}\n\n"
        "Give concise setup advice, identify missing fields, and suggest values that should be verified by the simulator."
    )
