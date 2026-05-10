"""
Streamlit frontend — same pipeline as notebooks (`System.analyze`).

Run from project root:
    python -m streamlit run app.py
"""
from __future__ import annotations

import io
import time

import streamlit as st
from PIL import Image

from streamlit_theme import inject_theme, render_agent_strip, render_hero
from streamlit_ui import render_analysis_results, render_disclaimer, render_sidebar
from system import System

SESSION_KEY = "analysis_result"
SESSION_ELAPSED = "analysis_elapsed_ms"
KEY_TEXT = "mh_user_text"
KEY_UPLOAD = "mh_face_upload"
KEY_CAM = "mh_face_cam"
SESSION_DIP_FP = "mh_dip_input_fingerprint"
SESSION_DIP_PREVIEW = "mh_dip_preview_result"
KEY_WEBCAM = "mh_webcam_enabled"

# Quick-fill demo texts so reviewers / users don't need to invent prompts.
SAMPLE_PROMPTS: list[tuple[str, str]] = [
    (
        "Stable / positive",
        "Today was a really good day. I went for a long walk in the morning, met my "
        "best friend for coffee, and finished a small project at work — feeling "
        "energised and grateful.",
    ),
    (
        "Negative / depressed",
        "I have been feeling completely exhausted and miserable for several weeks now. "
        "Nothing brings me joy anymore and even simple tasks feel impossible.",
    ),
    (
        "Mixed / monitor",
        "I keep telling everyone I am fine and trying to focus on small wins, but "
        "underneath I have been feeling tired, withdrawn and unmotivated for days.",
    ),
    (
        "Conflict (smiling, sad text)",
        "Outside everyone thinks I'm cheerful, but I have been crying myself to "
        "sleep most nights this month and I don't know why.",
    ),
    (
        "Crisis (override fires)",
        "I cannot do this anymore. I want to end my life tonight and nobody would "
        "even notice that I am gone.",
    ),
]


def _html(body: str) -> None:
    if hasattr(st, "html"):
        st.html(body)
    else:
        st.markdown(body, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading models…")
def get_system() -> System:
    return System()


def _clear_all() -> None:
    """Reset inputs + results.

    Must run via `on_click=` so it executes BEFORE the script reruns —
    Streamlit forbids writing to a widget's session_state key after the
    widget has already been instantiated in the current run.
    """
    for k in (
        SESSION_KEY,
        SESSION_ELAPSED,
        KEY_TEXT,
        KEY_UPLOAD,
        KEY_CAM,
        KEY_WEBCAM,
        SESSION_DIP_FP,
        SESSION_DIP_PREVIEW,
    ):
        st.session_state.pop(k, None)


def _fill_sample(text: str) -> None:
    """Pre-fill the text area with a sample prompt (called via on_click)."""
    st.session_state[KEY_TEXT] = text
    st.session_state.pop(SESSION_KEY, None)
    st.session_state.pop(SESSION_ELAPSED, None)


def _on_webcam_toggle() -> None:
    """Release camera widget state when turning webcam off; refresh DIP preview cache."""
    if not st.session_state.get(KEY_WEBCAM):
        st.session_state.pop(KEY_CAM, None)
    st.session_state.pop(SESSION_DIP_FP, None)
    st.session_state.pop(SESSION_DIP_PREVIEW, None)


def main() -> None:
    st.set_page_config(
        page_title="Multimodal Mental-Health Assistant",
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="🧠",
    )

    inject_theme()
    render_hero()

    try:
        system = get_system()
    except Exception as exc:  # noqa: BLE001
        render_agent_strip("none")
        st.warning(
            "Models are missing or failed to load. Train NLP + CNN (notebooks **02** and **03**), "
            "then refresh this page."
        )
        with st.expander("Technical details"):
            st.code(str(exc))
        render_disclaimer()
        return

    render_agent_strip(system.deepface.backend)
    render_sidebar(system)

    _html('<p class="mh-section-label">Inputs — text + face</p>')
    _html('<div class="mh-input-shell">')

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### ✍️ How are you feeling?")
        text = st.text_area(
            "text_input_main",
            key=KEY_TEXT,
            height=240,
            label_visibility="collapsed",
            placeholder=(
                "Write a few sentences (short slang words are unreliable for NLP). "
                "Example: I've felt exhausted and hopeless for weeks…"
            ),
        )
        with st.expander("✨ Try a sample prompt", expanded=False):
            st.caption(
                "Each prompt loads into the text box above. The crisis sample triggers "
                "the SafetyAgent override regardless of model probabilities."
            )
            sp_cols = st.columns(2)
            for i, (label, body) in enumerate(SAMPLE_PROMPTS):
                with sp_cols[i % 2]:
                    st.button(
                        label,
                        key=f"mh_sample_{i}",
                        on_click=_fill_sample,
                        args=(body,),
                        use_container_width=True,
                    )
    with col2:
        st.markdown("##### 📷 Face photo")
        uploaded = st.file_uploader(
            "Upload",
            key=KEY_UPLOAD,
            type=["png", "jpg", "jpeg", "webp"],
            help="Clear front-facing photo works best.",
        )
        st.toggle(
            "Webcam on",
            key=KEY_WEBCAM,
            help="Off = upload only (camera stays off). On = show capture widget.",
            on_change=_on_webcam_toggle,
        )
        cam = None
        if st.session_state.get(KEY_WEBCAM):
            cam = st.camera_input("Webcam", key=KEY_CAM)
        image: Image.Image | None = None
        raw_bytes: bytes | None = None
        if uploaded is not None:
            raw_bytes = uploaded.getvalue()
            image = Image.open(io.BytesIO(raw_bytes))
        elif cam is not None:
            raw_bytes = cam.getvalue()
            image = Image.open(io.BytesIO(raw_bytes))

        if image is not None and raw_bytes is not None:
            fp = hash(raw_bytes)
            if (
                st.session_state.get(SESSION_DIP_FP) != fp
                or SESSION_DIP_PREVIEW not in st.session_state
            ):
                with st.spinner("Applying DIP (face detect · CLAHE · denoise)…"):
                    st.session_state[SESSION_DIP_PREVIEW] = system.dip.preprocess(image)
                st.session_state[SESSION_DIP_FP] = fp
            dip_preview = st.session_state[SESSION_DIP_PREVIEW]

            pv_cols = st.columns(2)
            with pv_cols[0]:
                st.image(
                    dip_preview.original_image,
                    caption="Original input",
                    use_container_width=True,
                )
            with pv_cols[1]:
                st.image(
                    dip_preview.processed_image,
                    caption=(
                        "After DIP (face crop · CLAHE · denoise) — used when you Analyze"
                        if dip_preview.face_detected
                        else "After DIP (full frame · CLAHE · denoise) — used when you Analyze"
                    ),
                    use_container_width=True,
                )
            if dip_preview.note:
                st.caption(dip_preview.note)

    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns([1, 1, 1, 2])
    with btn_col1:
        analyze = st.button("⚡ Analyze", type="primary", use_container_width=True)
    with btn_col2:
        st.button(
            "Clear results",
            on_click=_clear_all,
            use_container_width=True,
        )
    with btn_col3:
        if st.button(
            "Clear cache",
            use_container_width=True,
            help=(
                "Clears Streamlit @st.cache_data / @st.cache_resource "
                "(reloads models + sidebar metrics on next run). Same idea as ⋮ menu → Clear cache."
            ),
        ):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.session_state.pop(SESSION_KEY, None)
            st.session_state.pop(SESSION_ELAPSED, None)
            st.rerun()
    with btn_col4:
        st.caption("Results stay visible until you clear — Streamlit reruns safely.")
    _html('</div>')

    if analyze:
        if not text or not text.strip():
            st.warning("Please enter some text first.")
        elif image is None:
            st.warning("Please upload or capture a face image first.")
        else:
            with st.spinner(
                "Running pipeline · NLP · CV · DeepFace · Fusion · Supervisor · XAI…"
            ):
                try:
                    t0 = time.perf_counter()
                    st.session_state[SESSION_KEY] = system.analyze(text, image)
                    st.session_state[SESSION_ELAPSED] = (time.perf_counter() - t0) * 1000.0
                except Exception as exc:  # noqa: BLE001
                    st.warning("Analysis could not finish. Check inputs and models, then try again.")
                    with st.expander("Details"):
                        st.code(str(exc))
                    st.session_state.pop(SESSION_KEY, None)
                    st.session_state.pop(SESSION_ELAPSED, None)

    if SESSION_KEY in st.session_state:
        _html('<div style="height:0.5rem"></div><p class="mh-section-label">Analysis output</p>')
        try:
            render_analysis_results(
                st.session_state[SESSION_KEY],
                elapsed_ms=st.session_state.get(SESSION_ELAPSED),
            )
        except Exception as exc:  # noqa: BLE001
            st.warning("Could not draw the full results panel. Try **Clear results** and run again.")
            with st.expander("Details"):
                st.code(str(exc))
    else:
        _html(
            '<div class="mh-empty-hint">'
            '<span class="icon">👆</span>'
            'Add <strong>text</strong> and a <strong>face image</strong>, '
            'then click <strong>Analyze</strong>.'
            '</div>'
        )

    render_disclaimer()


if __name__ == "__main__":
    main()
