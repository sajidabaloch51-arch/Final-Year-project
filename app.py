"""
IG-06 · Balochistan T2I Benchmark — Streamlit App
Browse dataset prompts and generate images using the same backends as server.py.

Run:  streamlit run app.py
"""
from __future__ import annotations

import base64
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from server import (  # noqa: E402
    _fresh_seed,
    build_image_prompt,
    generate_via_huggingface,
    generate_via_openai,
    generate_via_pollinations,
    generate_via_stable_horde,
)

PROMPTS_CSV = PROJECT_ROOT / "data" / "raw" / "prompts_with_balochi.csv"


@st.cache_data(show_spinner=False)
def load_prompts() -> pd.DataFrame:
    if not PROMPTS_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(PROMPTS_CSV)
    for col in ("prompt_id", "text", "label", "balochi_prompt"):
        if col not in df.columns:
            df[col] = ""
    return df


def data_uri_to_bytes(data_uri: str) -> bytes:
    if "," not in data_uri:
        return base64.b64decode(data_uri)
    return base64.b64decode(data_uri.split(",", 1)[1])


def generate_image(prompt: str, width: int, height: int, seed: int | None) -> tuple[str, str, int]:
    """Try providers in order; return (data_uri, provider, seed_used)."""
    seed_used = _fresh_seed(seed)
    image_prompt = build_image_prompt(prompt, seed=seed_used)
    errors: list[str] = []

    openai_key = os.environ.get("OPENAI_API_KEY")
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")

    if openai_key:
        try:
            return generate_via_openai(image_prompt, openai_key), "OpenAI DALL-E-3", seed_used
        except Exception as e:
            errors.append(f"OpenAI: {e}")

    if hf_token:
        try:
            return (
                generate_via_huggingface(image_prompt, hf_token),
                "Hugging Face Inference",
                seed_used,
            )
        except Exception as e:
            errors.append(f"HuggingFace: {e}")

    try:
        return (
            generate_via_pollinations(image_prompt, seed=seed_used, width=width, height=height),
            "Pollinations Flux",
            seed_used,
        )
    except Exception as e:
        errors.append(f"Pollinations: {e}")

    try:
        return (
            generate_via_stable_horde(image_prompt, seed=seed_used, width=width, height=height),
            "Stable Horde",
            seed_used,
        )
    except Exception as e:
        errors.append(f"Stable Horde: {e}")

    raise RuntimeError(" | ".join(errors) if errors else "All image providers failed.")


def main() -> None:
    st.set_page_config(
        page_title="Balochistan T2I Benchmark",
        page_icon="🖼️",
        layout="wide",
    )

    st.title("IG-06 · Balochistan Text-to-Image Benchmark")
    st.caption(
        "Browse dataset prompts and generate images. "
        "Your prompt text is used as-is (prompt-faithful generation)."
    )

    df = load_prompts()

    with st.sidebar:
        st.header("Settings")
        width = st.select_slider("Width", options=[512, 768, 1024], value=1024)
        height = st.select_slider("Height", options=[512, 768, 1024], value=1024)
        use_random_seed = st.checkbox("Random seed each run", value=True)
        fixed_seed = st.number_input("Fixed seed", min_value=1, max_value=2_147_483_647, value=42)
        st.divider()
        st.markdown("**Optional API keys** (env or `.env`)")
        st.code("OPENAI_API_KEY\nHF_TOKEN\nPOLLINATIONS_API_KEY\nSTABLE_HORDE_API_KEY", language="text")

    col_browse, col_gen = st.columns([1.1, 1])

    with col_browse:
        st.subheader("Prompt dataset")
        if df.empty:
            st.warning(f"No prompts found at `{PROMPTS_CSV}`.")
            selected_text = ""
            selected_balochi = ""
        else:
            categories = ["All"] + sorted(df["label"].dropna().astype(str).unique().tolist())
            category = st.selectbox("Category", categories)
            filtered = df if category == "All" else df[df["label"].astype(str) == category]
            search = st.text_input("Search prompts", "")
            if search.strip():
                q = search.strip().lower()
                mask = (
                    filtered["text"].astype(str).str.lower().str.contains(q, na=False)
                    | filtered["balochi_prompt"].astype(str).str.lower().str.contains(q, na=False)
                    | filtered["prompt_id"].astype(str).str.lower().str.contains(q, na=False)
                )
                filtered = filtered[mask]

            st.write(f"{len(filtered):,} prompts")
            options = filtered.head(300).reset_index(drop=True)
            labels = [
                f"{r.prompt_id} · {r.label} · {str(r.text)[:80]}"
                for r in options.itertuples(index=False)
            ]
            choice = st.selectbox("Select a prompt", ["(custom / none)"] + labels)
            selected_text = ""
            selected_balochi = ""
            if choice != "(custom / none)":
                idx = labels.index(choice)
                row = options.iloc[idx]
                selected_text = str(row.get("text") or "")
                selected_balochi = str(row.get("balochi_prompt") or "")
                st.success(f"Selected `{row['prompt_id']}` ({row['label']})")
                if st.button("Load selected into generator", use_container_width=True):
                    st.session_state["prompt_input"] = selected_text
                    st.rerun()

            if selected_balochi and selected_balochi != "nan":
                with st.expander("Balochi prompt"):
                    st.write(selected_balochi)
                    if st.button("Load Balochi into generator", use_container_width=True):
                        st.session_state["prompt_input"] = selected_balochi
                        st.rerun()

    with col_gen:
        st.subheader("Generate image")
        if "prompt_input" not in st.session_state:
            st.session_state["prompt_input"] = selected_text or ""
        prompt = st.text_area(
            "Image prompt",
            height=160,
            placeholder="Enter or select a prompt…",
            key="prompt_input",
        )

        generate = st.button("Generate Image", type="primary", use_container_width=True)

        if generate:
            if not prompt.strip():
                st.error("Please enter or select a prompt first.")
            else:
                seed_arg = None if use_random_seed else int(fixed_seed)
                with st.spinner("Generating image from your prompt…"):
                    try:
                        data_uri, provider, seed_used = generate_image(
                            prompt.strip(), width=int(width), height=int(height), seed=seed_arg
                        )
                        img_bytes = data_uri_to_bytes(data_uri)
                        st.session_state["last_image"] = img_bytes
                        st.session_state["last_provider"] = provider
                        st.session_state["last_seed"] = seed_used
                        st.session_state["last_prompt"] = prompt.strip()
                    except Exception as e:
                        st.error(f"Generation failed: {e}")

        if st.session_state.get("last_image"):
            st.image(st.session_state["last_image"], use_container_width=True)
            st.caption(
                f"Provider: {st.session_state.get('last_provider')} · "
                f"seed {st.session_state.get('last_seed')}"
            )
            st.download_button(
                "Download image",
                data=st.session_state["last_image"],
                file_name="balochistan_t2i.png",
                mime="image/png",
                use_container_width=True,
            )

    st.divider()
    st.markdown(
        "Also available: FastAPI dashboard via `python server.py` → http://127.0.0.1:8000 · "
        "Full benchmark via `python main.py`"
    )


if __name__ == "__main__":
    main()
