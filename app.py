"""Streamlit programa trumpų vaizdo klipų highlight reel generavimui su Gemini."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
import time
import hmac
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Literal

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from moviepy import (
    AudioFileClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    VideoFileClip,
    concatenate_audioclips,
    concatenate_videoclips,
)
from moviepy.video import fx as vfx

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_MODEL_OPTIONS: dict[str, str] = {
    "Gemini 2.5 Flash – greitas": "gemini-2.5-flash",
    "Gemini 2.5 Pro – tikslus": "gemini-2.5-pro",
}
DEFAULT_GEMINI_MODEL_LABEL = "Gemini 2.5 Flash – greitas"
DEFAULT_APP_USERNAME = "pmc_admin"
DEFAULT_APP_PASSWORD = "Saule2007"
MAX_CLIP_DURATION_SEC = 60.0
DEFAULT_BACKGROUND_VOLUME = 0.3
SPEECH_DUCK_VOLUME = 0.1
OFFICIAL_SPEECHES_KEY = "official_speeches"
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi"}
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
SUPPORTED_EXTENSIONS = SUPPORTED_VIDEO_EXTENSIONS
SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".wav"}
SUPPORTED_LOGO_EXTENSIONS = {".png", ".jpg", ".jpeg"}
SUPPORTED_MEDIA_EXTENSIONS = SUPPORTED_VIDEO_EXTENSIONS | SUPPORTED_IMAGE_EXTENSIONS
IMAGE_CLIP_DURATION_SEC = 4.0
DEFAULT_CLIP_FPS = 24
LOGO_HEIGHT_PX = 50
LOGO_MARGIN_PX = 16
MIME_TYPES = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
}

HIGHLIGHT_JSON_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "start": {"type": "number", "description": "Pradžios laikas sekundėmis"},
            "end": {"type": "number", "description": "Pabaigos laikas sekundėmis"},
            "reason": {"type": "string", "description": "Kodėl šis momentas pasirinktas"},
            "is_official_speech": {
                "type": "boolean",
                "description": "True jei tai Direktoriaus ar Pavaduotojos oficiali kalba",
            },
        },
        "required": ["start", "end", "reason"],
    },
}


def build_highlight_json_schema(include_speech_flag: bool) -> dict[str, Any]:
    if not include_speech_flag:
        return HIGHLIGHT_JSON_SCHEMA

    schema = json.loads(json.dumps(HIGHLIGHT_JSON_SCHEMA))
    schema["items"]["required"] = ["start", "end", "reason", "is_official_speech"]
    return schema


@dataclass(frozen=True)
class HighlightSegment:
    start: float
    end: float
    reason: str
    is_official_speech: bool = False


@dataclass(frozen=True)
class UploadedMediaItem:
    path: Path
    kind: Literal["video", "image"]


class AppError(Exception):
    """Naudotojui rodoma klaida."""


def resolve_gemini_model(model_label: str) -> str:
    model_id = GEMINI_MODEL_OPTIONS.get(model_label)
    if model_id is None:
        raise AppError(f"Nežinomas Gemini modelis: {model_label}")
    return model_id


def get_auth_credentials() -> tuple[str, str]:
    username = os.getenv("APP_USERNAME", DEFAULT_APP_USERNAME).strip()
    password = os.getenv("APP_PASSWORD", DEFAULT_APP_PASSWORD).strip()
    if not username or not password:
        raise AppError("APP_USERNAME ir APP_PASSWORD turi būti nustatyti .env faile.")
    return username, password


def credentials_valid(username: str, password: str) -> bool:
    expected_username, expected_password = get_auth_credentials()
    username_match = hmac.compare_digest(
        username.strip().casefold(),
        expected_username.casefold(),
    )
    password_match = hmac.compare_digest(password.strip(), expected_password)
    return username_match and password_match


def logout_user() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state["authenticated"] = False
    st.rerun()


def get_gemini_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise AppError(
            "Nerastas GEMINI_API_KEY. Užpildykite .env failą arba nustatykite aplinkos kintamąjį."
        )
    return genai.Client(api_key=api_key)


def is_supported_video(filename: str) -> bool:
    return Path(filename).suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS


def is_supported_image(filename: str) -> bool:
    return Path(filename).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def is_supported_media(filename: str) -> bool:
    return Path(filename).suffix.lower() in SUPPORTED_MEDIA_EXTENSIONS


def get_mime_type(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    mime_type = MIME_TYPES.get(suffix)
    if mime_type is None:
        raise AppError(f"Nepalaikomas vaizdo formatas: {suffix}")
    return mime_type


def wait_for_file_active(client: genai.Client, file_name: str, timeout_sec: int = 300) -> None:
    deadline = time.time() + timeout_sec
    current = client.files.get(name=file_name)

    while _file_state_name(current) != "ACTIVE":
        state = _file_state_name(current)
        if state == "FAILED":
            raise AppError("Gemini nepavyko apdoroti įkeltos vaizdo bylos.")

        if time.time() >= deadline:
            raise AppError("Baigėsi laikas laukti, kol vaizdo failas bus paruoštas Gemini.")

        time.sleep(2)
        current = client.files.get(name=file_name)


def _file_state_name(file_obj: Any) -> str:
    state = getattr(file_obj, "state", None)
    if state is None:
        return "UNKNOWN"
    if isinstance(state, str):
        return state
    return str(getattr(state, "name", state))


def build_analysis_prompt(
    criteria: str,
    duration_sec: float,
    *,
    include_speech_flag: bool,
) -> str:
    prompt = (
        "Analizuok šį vaizdo įrašą ir parink geriausius momentus pagal šiuos kriterijus:\n"
        f"{criteria.strip()}\n\n"
        "Grąžink tik JSON masyvą be jokio papildomo teksto. "
        "Kiekvienas objektas turi laukus: start (float, sek.), end (float, sek.), reason (string)."
    )

    if include_speech_flag:
        prompt += (
            " Taip pat privalomas laukas is_official_speech (boolean): "
            "nustatyk true, jei segmente Direktorius ar Pavaduotoja kalba oficialiai; "
            "kitu atveju false."
        )

    prompt += (
        f"\nVaizdo trukmė: {duration_sec:.2f} sek. "
        f"Visi intervalai turi būti 0 <= start < end <= {duration_sec:.2f}. "
        "Pasirink 1–5 geriausius momentus. Jei tinkamų momentų nėra, grąžink tuščią masyvą []."
    )
    return prompt


def parse_highlights(raw_text: str, video_duration: float) -> list[HighlightSegment]:
    text = raw_text.strip()
    if not text:
        raise AppError("Gemini grąžino tuščią atsakymą.")

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", text)
        if not match:
            raise AppError("Nepavyko išanalizuoti Gemini JSON atsakymo.") from None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise AppError("Nepavyko išanalizuoti Gemini JSON atsakymo.") from exc

    if not isinstance(payload, list):
        raise AppError("Gemini atsakymas turi būti JSON masyvas.")

    highlights: list[HighlightSegment] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            logger.warning("Praleidžiamas netinkamo formato segmentas #%s", index)
            continue

        try:
            start = float(item["start"])
            end = float(item["end"])
            reason = str(item.get("reason", "")).strip()
            is_official_speech = bool(item.get("is_official_speech", False))
        except (KeyError, TypeError, ValueError):
            logger.warning("Praleidžiamas netinkamas segmentas #%s", index)
            continue

        if start < 0 or end <= start or end > video_duration + 0.05:
            logger.warning(
                "Praleidžiamas segmentas už vaizdo ribų: %.2f–%.2f (trukmė %.2f)",
                start,
                end,
                video_duration,
            )
            continue

        highlights.append(
            HighlightSegment(
                start=max(0.0, start),
                end=min(end, video_duration),
                reason=reason or "Pasirinktas pagal kriterijus",
                is_official_speech=is_official_speech,
            )
        )

    return highlights


def analyze_video_with_gemini(
    client: genai.Client,
    video_path: Path,
    criteria: str,
    model_id: str,
    *,
    include_speech_flag: bool = False,
) -> list[HighlightSegment]:
    uploaded = None
    try:
        uploaded = client.files.upload(
            file=str(video_path),
            config={"display_name": video_path.name},
        )
        wait_for_file_active(client, uploaded.name)

        with VideoFileClip(str(video_path)) as probe_clip:
            duration = float(probe_clip.duration or 0.0)

        if duration <= 0:
            raise AppError(f"Nepavyko nustatyti vaizdo trukmės: {video_path.name}")

        prompt = build_analysis_prompt(
            criteria,
            duration,
            include_speech_flag=include_speech_flag,
        )
        response = client.models.generate_content(
            model=model_id,
            contents=[uploaded, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=build_highlight_json_schema(include_speech_flag),
                temperature=0.2,
            ),
        )

        raw_text = (response.text or "").strip()
        return parse_highlights(raw_text, duration)
    finally:
        if uploaded is not None:
            try:
                client.files.delete(name=uploaded.name)
            except Exception:
                logger.exception("Nepavyko ištrinti Gemini failo: %s", uploaded.name)


def close_clip(clip: Any | None) -> None:
    if clip is not None:
        try:
            clip.close()
        except Exception:
            logger.exception("Klaida uždarant MoviePy klipą")


def close_clips(clips: list[Any]) -> None:
    for clip in clips:
        close_clip(clip)


def extract_segments_from_video(
    video_path: Path,
    highlights: list[HighlightSegment],
    *,
    track_official_speeches: bool = False,
) -> tuple[list[VideoFileClip], VideoFileClip, list[tuple[float, float]]]:
    parent_clip = VideoFileClip(str(video_path))
    subclips: list[VideoFileClip] = []
    speech_intervals: list[tuple[float, float]] = []
    timeline_offset = 0.0

    for segment in highlights:
        subclip = parent_clip.subclipped(segment.start, segment.end)
        clip_duration = float(subclip.duration or 0.0)

        if track_official_speeches and segment.is_official_speech and clip_duration > 0:
            speech_intervals.append((timeline_offset, timeline_offset + clip_duration))

        timeline_offset += clip_duration
        subclips.append(subclip)

    return subclips, parent_clip, speech_intervals


def save_uploaded_media(
    uploaded_files: list[Any],
    work_dir: Path,
) -> list[UploadedMediaItem]:
    saved_items: list[UploadedMediaItem] = []

    for uploaded in uploaded_files:
        suffix = Path(uploaded.name).suffix.lower()
        if suffix not in SUPPORTED_MEDIA_EXTENSIONS:
            raise AppError(
                f"Nepalaikomas formatas: {uploaded.name}. "
                "Leidžiami: mp4, mov, avi, png, jpg, jpeg."
            )

        destination = work_dir / uploaded.name
        with open(destination, "wb") as handle:
            handle.write(uploaded.getbuffer())

        if suffix in SUPPORTED_VIDEO_EXTENSIONS:
            with VideoFileClip(str(destination)) as clip:
                duration = float(clip.duration or 0.0)

            if duration > MAX_CLIP_DURATION_SEC:
                raise AppError(
                    f"{uploaded.name} per ilgas ({duration:.1f} s). "
                    f"Maksimali trukmė: {int(MAX_CLIP_DURATION_SEC)} s."
                )

            saved_items.append(UploadedMediaItem(path=destination, kind="video"))
            continue

        saved_items.append(UploadedMediaItem(path=destination, kind="image"))

    return saved_items


def create_animated_image_clip(image_path: Path) -> VideoFileClip:
    duration = IMAGE_CLIP_DURATION_SEC
    clip = (
        ImageClip(str(image_path))
        .with_duration(duration)
        .with_fps(DEFAULT_CLIP_FPS)
    )

    if clip.w > 1920:
        clip = clip.resized(width=1920)

    def zoom_factor(t: float) -> float:
        progress = min(max(t / duration, 0.0), 1.0)
        return 1.0 + 0.08 * progress

    return clip.with_effects(
        [
            vfx.Resize(zoom_factor),
            vfx.FadeIn(0.4),
            vfx.FadeOut(0.4),
        ]
    )


def save_uploaded_audio(uploaded_file: Any, work_dir: Path) -> Path:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in SUPPORTED_AUDIO_EXTENSIONS:
        raise AppError(
            f"Nepalaikomas garso formatas: {uploaded_file.name}. Leidžiami: mp3, wav."
        )

    destination = work_dir / uploaded_file.name
    with open(destination, "wb") as handle:
        handle.write(uploaded_file.getbuffer())

    with AudioFileClip(str(destination)) as probe_audio:
        duration = float(probe_audio.duration or 0.0)

    if duration <= 0:
        raise AppError(f"Nepavyko nuskaityti fono muzikos trukmės: {uploaded_file.name}")

    return destination


def save_uploaded_logo(uploaded_file: Any, work_dir: Path) -> Path:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in SUPPORTED_LOGO_EXTENSIONS:
        raise AppError(
            f"Nepalaikomas logotipo formatas: {uploaded_file.name}. Leidžiami: png, jpg, jpeg."
        )

    destination = work_dir / uploaded_file.name
    with open(destination, "wb") as handle:
        handle.write(uploaded_file.getbuffer())

    return destination


def apply_logo_overlay(
    video_clip: VideoFileClip,
    logo_path: Path,
) -> tuple[VideoFileClip, list[Any]]:
    managed_clips: list[Any] = []
    duration = float(video_clip.duration or 0.0)

    logo_clip = ImageClip(str(logo_path)).with_duration(duration)
    managed_clips.append(logo_clip)

    resized_logo = logo_clip.resized(height=LOGO_HEIGHT_PX)
    managed_clips.append(resized_logo)

    x_pos = max(video_clip.w - resized_logo.w - LOGO_MARGIN_PX, LOGO_MARGIN_PX)
    positioned_logo = resized_logo.with_position((x_pos, LOGO_MARGIN_PX))

    composited = CompositeVideoClip([video_clip, positioned_logo])
    managed_clips.append(composited)
    return composited, managed_clips


def fit_audio_to_duration(audio_clip: AudioFileClip, target_duration: float) -> AudioFileClip:
    if target_duration <= 0:
        raise AppError("Galutinio vaizdo trukmė turi būti didesnė už nulį.")

    source_duration = float(audio_clip.duration or 0.0)
    if source_duration <= 0:
        raise AppError("Fono muzikos failo trukmė nerasta.")

    if source_duration >= target_duration:
        return audio_clip.subclipped(0, target_duration)

    loops_needed = int(target_duration // source_duration) + 1
    looped = concatenate_audioclips([audio_clip] * loops_needed)
    return looped.subclipped(0, target_duration)


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []

    normalized = sorted(
        (max(0.0, start), max(start, end))
        for start, end in intervals
        if end > start
    )
    if not normalized:
        return []

    merged: list[tuple[float, float]] = [normalized[0]]
    for start, end in normalized[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def build_volume_shaped_background(
    fitted_audio: AudioFileClip,
    total_duration: float,
    speech_intervals: list[tuple[float, float]],
    normal_volume: float,
    speech_volume: float,
    managed_clips: list[Any],
) -> AudioFileClip:
    intervals = merge_intervals(speech_intervals)
    if not intervals:
        scaled = fitted_audio.with_volume_scaled(normal_volume)
        if scaled is not fitted_audio:
            managed_clips.append(scaled)
        return scaled

    pieces: list[AudioFileClip] = []
    cursor = 0.0

    for speech_start, speech_end in intervals:
        clamped_start = min(max(0.0, speech_start), total_duration)
        clamped_end = min(max(clamped_start, speech_end), total_duration)

        if cursor < clamped_start:
            normal_chunk = fitted_audio.subclipped(cursor, clamped_start).with_volume_scaled(
                normal_volume
            )
            pieces.append(normal_chunk)
            managed_clips.append(normal_chunk)

        if clamped_end > clamped_start:
            speech_chunk = fitted_audio.subclipped(clamped_start, clamped_end).with_volume_scaled(
                speech_volume
            )
            pieces.append(speech_chunk)
            managed_clips.append(speech_chunk)

        cursor = clamped_end

    if cursor < total_duration:
        tail_chunk = fitted_audio.subclipped(cursor, total_duration).with_volume_scaled(
            normal_volume
        )
        pieces.append(tail_chunk)
        managed_clips.append(tail_chunk)

    if len(pieces) == 1:
        return pieces[0]

    shaped = concatenate_audioclips(pieces)
    managed_clips.append(shaped)
    return shaped


def apply_background_music(
    video_clip: VideoFileClip,
    music_path: Path,
    *,
    mix_with_original: bool,
    volume: float,
    speech_intervals: list[tuple[float, float]] | None = None,
    duck_speech_volume: float = SPEECH_DUCK_VOLUME,
) -> tuple[VideoFileClip, list[Any]]:
    managed_clips: list[Any] = []
    background_audio = AudioFileClip(str(music_path))
    managed_clips.append(background_audio)

    total_duration = float(video_clip.duration or 0.0)
    fitted_audio = fit_audio_to_duration(background_audio, total_duration)
    if fitted_audio is not background_audio:
        managed_clips.append(fitted_audio)

    if speech_intervals:
        scaled_audio = build_volume_shaped_background(
            fitted_audio,
            total_duration,
            speech_intervals,
            normal_volume=volume,
            speech_volume=duck_speech_volume,
            managed_clips=managed_clips,
        )
    else:
        scaled_audio = fitted_audio.with_volume_scaled(volume)
        if scaled_audio is not fitted_audio:
            managed_clips.append(scaled_audio)

    if mix_with_original and video_clip.audio is not None:
        mixed_audio = CompositeAudioClip([video_clip.audio, scaled_audio])
        managed_clips.append(mixed_audio)
        return video_clip.with_audio(mixed_audio), managed_clips

    return video_clip.with_audio(scaled_audio), managed_clips


def build_highlight_reel(
    media_items: list[UploadedMediaItem],
    criteria: str,
    model_id: str,
    progress_bar: Any,
    status_box: Any,
    background_music_path: Path | None = None,
    mix_with_original: bool = True,
    background_volume: float = DEFAULT_BACKGROUND_VOLUME,
    prioritize_official_speeches: bool = False,
    logo_path: Path | None = None,
) -> Path:
    client = get_gemini_client()
    all_subclips: list[VideoFileClip] = []
    parent_clips: list[VideoFileClip] = []
    speech_intervals: list[tuple[float, float]] = []
    audio_clips: list[Any] = []
    overlay_clips: list[Any] = []
    final_clip: VideoFileClip | None = None
    output_path = media_items[0].path.parent / "highlight_reel.mp4"
    include_speech_flag = prioritize_official_speeches
    duck_music_for_speeches = (
        prioritize_official_speeches and background_music_path is not None
    )
    global_timeline_offset = 0.0

    total_steps = len(media_items) + 1

    try:
        for index, media_item in enumerate(media_items, start=1):
            progress_bar.progress(int(((index - 1) / total_steps) * 100))

            if media_item.kind == "image":
                status_box.info(
                    f"Ruošiamas paveikslėlis {index}/{len(media_items)}: `{media_item.path.name}`"
                )
                image_clip = create_animated_image_clip(media_item.path)
                clip_duration = float(image_clip.duration or 0.0)
                global_timeline_offset += clip_duration
                all_subclips.append(image_clip)
                continue

            video_path = media_item.path
            status_box.info(
                f"Analizuojamas vaizdas {index}/{len(media_items)}: `{video_path.name}`"
            )

            highlights = analyze_video_with_gemini(
                client,
                video_path,
                criteria,
                model_id,
                include_speech_flag=include_speech_flag,
            )
            if not highlights:
                logger.info("Vaizde %s nerasta tinkamų segmentų.", video_path.name)
                continue

            subclips, parent, local_speech_intervals = extract_segments_from_video(
                video_path,
                highlights,
                track_official_speeches=duck_music_for_speeches,
            )
            for local_start, local_end in local_speech_intervals:
                speech_intervals.append(
                    (global_timeline_offset + local_start, global_timeline_offset + local_end)
                )

            global_timeline_offset += sum(float(clip.duration or 0.0) for clip in subclips)
            all_subclips.extend(subclips)
            parent_clips.append(parent)

        if not all_subclips:
            raise AppError(
                "Nepavyko surinkti nė vieno tinkamo momento. Pabandykite kitus kriterijus ar vaizdo įrašus."
            )

        progress_bar.progress(int((len(media_items) / total_steps) * 100))
        status_box.info("Sujungiami pasirinkti klipai į galutinį vaizdo įrašą...")

        final_clip = concatenate_videoclips(all_subclips, method="compose")

        if background_music_path is not None:
            if duck_music_for_speeches and speech_intervals:
                status_box.info(
                    "Pridedama fono muzika – oficialių kalbų metu garsumas sumažinamas iki 10%..."
                )
            else:
                status_box.info("Pridedama fono muzika prie galutinio vaizdo įrašo...")

            final_clip, music_clips = apply_background_music(
                final_clip,
                background_music_path,
                mix_with_original=mix_with_original,
                volume=background_volume,
                speech_intervals=speech_intervals if duck_music_for_speeches else None,
                duck_speech_volume=SPEECH_DUCK_VOLUME,
            )
            audio_clips.extend(music_clips)

        if logo_path is not None:
            status_box.info("Pridedamas logotipas ant galutinio vaizdo įrašo...")
            final_clip, logo_clips = apply_logo_overlay(final_clip, logo_path)
            overlay_clips.extend(logo_clips)

        final_clip.write_videofile(
            str(output_path),
            codec="libx264",
            audio_codec="aac",
            logger=None,
        )

        progress_bar.progress(100)
        status_box.success("Vaizdo montažas paruoštas!")
        return output_path
    finally:
        close_clip(final_clip)
        close_clips(audio_clips)
        close_clips(overlay_clips)
        for subclip in all_subclips:
            close_clip(subclip)
        for parent in parent_clips:
            close_clip(parent)


def cleanup_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


@dataclass(frozen=True)
class CriteriaPreset:
    key: str
    label: str
    descriptor: str


CRITERIA_PRESETS: tuple[CriteriaPreset, ...] = (
    CriteriaPreset(
        key="smiles",
        label="Šypsenos ir juokas",
        descriptor="Focus on smiles, laughter, and joyful expressions.",
    ),
    CriteriaPreset(
        key="family",
        label="Šilti šeimos momentai ir apsikabinimai",
        descriptor="Prioritize warm family moments, togetherness, and hugs.",
    ),
    CriteriaPreset(
        key="action",
        label="Aktyvus veiksmas ir judesys",
        descriptor="Select active scenes with strong movement, energy, and action.",
    ),
    CriteriaPreset(
        key="scenic",
        label="Gražūs gamtos vaizdai ir saulėlydžiai",
        descriptor="Highlight beautiful nature shots, landscapes, and sunsets.",
    ),
    CriteriaPreset(
        key="fast_paced",
        label="Greiti, dinamiški kadrai (trumpo formato stilius)",
        descriptor="Prefer fast-paced, dynamic shots suitable for short-form reels.",
    ),
    CriteriaPreset(
        key="exclude_operator_voice",
        label="Pašalinti filmuotojo balsą / komentarus fone",
        descriptor=(
            "CRITICAL AUDIO INSTRUCTION: Identify any timestamps where the person filming "
            "the video (the camera operator) is speaking, giving instructions, or making "
            "background commentary. Do NOT include these timestamps in the final highlights. "
            "We only want moments with natural background sounds or where the main subjects "
            "of the video are talking/interacting, completely excluding the operator's voice."
        ),
    ),
    CriteriaPreset(
        key=OFFICIAL_SPEECHES_KEY,
        label="Išsaugoti oficialias kalbas (Direktoriaus / Pavaduotojos)",
        descriptor=(
            "PRIORITY CONTENT INSTRUCTION: Look for segments where the Director (Direktorius) "
            "or Deputy Director (Pavaduotoja) is giving an official speech or address. "
            "These segments are highly important. Identify their speeches, and make sure to "
            "include them cleanly in the highlights. Do NOT cut them mid-sentence; ensure the "
            "timestamps capture the full context of their speech from start to finish so it "
            "sounds and looks professional in the final montage."
        ),
    ),
)


def build_criteria_descriptor(selected_keys: list[str], custom_notes: str) -> str:
    parts: list[str] = []

    for preset in CRITERIA_PRESETS:
        if preset.key in selected_keys:
            parts.append(preset.descriptor)

    if custom_notes.strip():
        parts.append(custom_notes.strip())

    if not parts:
        return ""

    return "\n\n".join(parts)


def configure_page() -> None:
    st.set_page_config(
        page_title="PMC Vaizdo intelektas",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

        :root {
            --pmc-bg-deep: #0A0F1D;
            --pmc-navy: #0A192F;
            --pmc-red: #DC2626;
            --pmc-red-hover: #B91C1C;
            --pmc-silver: #CBD5E1;
            --pmc-slate: #64748B;
            --pmc-white: #FFFFFF;
            --pmc-glass: rgba(30, 41, 59, 0.7);
            --pmc-glass-border: rgba(148, 163, 184, 0.18);
            --pmc-glass-hover: rgba(30, 41, 59, 0.85);
            --primary-color: #DC2626;
            --primary-color-hover: #B91C1C;
        }

        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"],
        section.main {
            --primary-color: #DC2626 !important;
            --primary-color-hover: #B91C1C !important;
        }

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', sans-serif;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
            color: var(--pmc-silver);
        }

        /* ── Dark ambient glow background ── */
        .stApp,
        [data-testid="stAppViewContainer"],
        section.main > div {
            background-color: #0A0F1D !important;
            background-image:
                radial-gradient(ellipse 85% 70% at 0% 45%, rgba(37, 99, 235, 0.22) 0%, transparent 58%),
                radial-gradient(ellipse 75% 65% at 100% 55%, rgba(185, 28, 28, 0.18) 0%, transparent 52%),
                radial-gradient(ellipse 50% 40% at 15% 85%, rgba(79, 70, 229, 0.12) 0%, transparent 55%),
                radial-gradient(ellipse 45% 35% at 88% 12%, rgba(127, 29, 29, 0.14) 0%, transparent 50%),
                linear-gradient(165deg, #0A0F1D 0%, #0D1424 48%, #0A0F1D 100%) !important;
        }

        header[data-testid="stHeader"] {
            background: transparent !important;
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
        }

        div[data-testid="stToolbar"] {
            background: transparent !important;
        }

        div[data-testid="stToolbar"] button,
        div[data-testid="stToolbar"] span,
        div[data-testid="stToolbar"] svg {
            color: #CBD5E1 !important;
            fill: #CBD5E1 !important;
        }

        .block-container {
            color: #CBD5E1;
        }

        .block-container h1,
        .block-container h2,
        .block-container h3,
        .block-container h4,
        .block-container strong {
            color: #FFFFFF !important;
        }

        .block-container label,
        .block-container p,
        .block-container span,
        .block-container li {
            color: #CBD5E1;
        }

        div[data-testid="stSidebar"] {
            background: rgba(15, 23, 42, 0.72);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-right: 1px solid var(--pmc-glass-border);
        }

        div[data-testid="stSidebar"] .block-container {
            padding-top: 1.5rem;
            color: #CBD5E1;
        }

        div[data-testid="stSidebar"] .stMarkdown,
        div[data-testid="stSidebar"] label,
        div[data-testid="stSidebar"] p,
        div[data-testid="stSidebar"] span {
            color: #CBD5E1;
        }

        /* ── Hero banner ── */
        .hero-wrap {
            text-align: center;
            padding: 2.4rem 1.75rem 2rem;
            margin-bottom: 1rem;
            border-radius: 12px;
            background: rgba(30, 41, 59, 0.7) !important;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid var(--pmc-glass-border);
            box-shadow:
                0 10px 40px rgba(0, 0, 0, 0.35),
                inset 0 1px 0 rgba(255, 255, 255, 0.04);
        }

        .hero-emoji {
            font-size: 2.4rem;
            line-height: 1;
            margin-bottom: 0.75rem;
        }

        .hero-wrap .hero-brand,
        span.hero-brand {
            display: inline-block !important;
            padding: 0.28rem 0.85rem !important;
            margin-right: 0.55rem !important;
            border-radius: 8px !important;
            background: #DC2626 !important;
            background-color: #DC2626 !important;
            border: 2px solid #DC2626 !important;
            color: #FFFFFF !important;
            -webkit-text-fill-color: #FFFFFF !important;
            font-size: clamp(1.35rem, 3vw, 1.9rem) !important;
            font-weight: 800 !important;
            letter-spacing: 0.1em !important;
            vertical-align: middle !important;
            box-shadow: 0 6px 18px rgba(220, 38, 38, 0.45) !important;
        }

        .hero-title {
            font-size: clamp(1.75rem, 4vw, 2.45rem);
            font-weight: 800;
            letter-spacing: -0.02em;
            margin: 0 0 0.5rem 0;
            color: #FFFFFF !important;
        }

        .hero-title-accent {
            color: #FFFFFF !important;
            -webkit-text-fill-color: #FFFFFF !important;
            background: none !important;
        }

        .hero-title-sub {
            display: block;
            margin-top: 0.25rem;
            margin-bottom: 0.85rem;
            font-size: clamp(1rem, 2.2vw, 1.25rem);
            font-weight: 600;
            color: #CBD5E1;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }

        .hero-subtitle {
            font-size: 1.02rem;
            color: #E2E8F0;
            max-width: 760px;
            margin: 0 auto;
            line-height: 1.7;
        }

        /* ── Cards ── */
        .panel-card {
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid var(--pmc-glass-border);
            border-radius: 12px;
            padding: 1.25rem 1.35rem 0.35rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.28);
            margin-bottom: 1rem;
        }

        .panel-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #FFFFFF;
            margin-bottom: 0.35rem;
            border-left: 3px solid #DC2626;
            padding-left: 0.65rem;
        }

        .panel-caption {
            color: #94A3B8;
            font-size: 0.92rem;
            margin-bottom: 0.85rem;
        }

        .success-card {
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid var(--pmc-glass-border);
            border-left: 4px solid #DC2626;
            border-radius: 12px;
            padding: 1.5rem 1.6rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.28);
            margin: 1rem 0 1.25rem;
        }

        .success-title {
            font-size: 1.35rem;
            font-weight: 800;
            color: #FFFFFF;
            margin: 0 0 0.35rem 0;
        }

        .success-text {
            color: #CBD5E1;
            margin: 0 0 1rem 0;
        }

        .sidebar-badge {
            display: inline-block;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            background: rgba(220, 38, 38, 0.12);
            border: 1px solid #DC2626;
            color: #DC2626 !important;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 0.75rem;
        }

        .pmc-sidebar-logo {
            font-size: 1.45rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            color: #FFFFFF;
            margin-bottom: 0.15rem;
        }

        div[data-testid="stSidebar"] h3 {
            color: #FFFFFF;
            border-left: 3px solid #DC2626;
            padding-left: 0.55rem;
        }

        .pmc-sidebar-tagline {
            color: #94A3B8;
            font-size: 0.82rem;
            margin-bottom: 1rem;
        }

        .criteria-preview {
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid var(--pmc-glass-border);
            border-radius: 12px;
            padding: 0.85rem 0.95rem;
            color: #CBD5E1;
            font-size: 0.88rem;
            line-height: 1.55;
            margin-top: 0.5rem;
        }

        /* ── Form controls ── */
        div[data-testid="stFileUploader"] section {
            border: 1.5px dashed rgba(148, 163, 184, 0.35) !important;
            border-radius: 12px !important;
            background: rgba(30, 41, 59, 0.7) !important;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            padding: 0.35rem !important;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.22) !important;
        }

        div[data-testid="stFileUploader"] section:hover {
            border-color: #DC2626 !important;
        }

        div[data-testid="stFileUploader"] section span,
        div[data-testid="stFileUploader"] section small,
        div[data-testid="stFileUploader"] section p {
            color: #CBD5E1 !important;
        }

        div[data-testid="stTextArea"] textarea,
        div[data-testid="stTextInput"] input {
            border: none !important;
            box-shadow: none !important;
            outline: none !important;
            background: transparent !important;
            color: #FFFFFF !important;
        }

        div[data-testid="stTextArea"] textarea::placeholder,
        div[data-testid="stTextInput"] input::placeholder {
            color: #64748B !important;
        }

        div[data-testid="stTextArea"] > div > div,
        div[data-testid="stTextInput"] > div > div,
        div[data-testid="stTextInput"] [data-baseweb="input"],
        div[data-testid="stTextArea"] [data-baseweb="textarea"] {
            border-radius: 12px !important;
            border: 1px solid var(--pmc-glass-border) !important;
            background: rgba(30, 41, 59, 0.7) !important;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
        }

        div[data-testid="stTextArea"]:focus-within > div > div,
        div[data-testid="stTextInput"]:focus-within > div > div,
        div[data-testid="stTextInput"]:focus-within [data-baseweb="input"],
        div[data-testid="stTextArea"]:focus-within [data-baseweb="textarea"] {
            border-color: #DC2626 !important;
            box-shadow: 0 0 0 2px rgba(220, 38, 38, 0.2) !important;
        }

        div[data-testid="InputInstructions"] {
            display: none !important;
        }

        div[data-testid="stSelectbox"] > div > div,
        div[data-testid="stMultiSelect"] > div > div {
            background: rgba(30, 41, 59, 0.7) !important;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid var(--pmc-glass-border) !important;
            border-radius: 12px !important;
            color: #FFFFFF !important;
        }

        div[data-testid="stSelectbox"] label,
        div[data-testid="stMultiSelect"] label,
        div[data-testid="stSlider"] label {
            color: #CBD5E1 !important;
        }

        a {
            color: #CBD5E1;
            transition: color 0.2s ease;
        }

        a:hover {
            color: #DC2626 !important;
        }

        div[data-testid="stSidebar"] .stButton > button {
            background: rgba(30, 41, 59, 0.7) !important;
            border: 1px solid var(--pmc-glass-border) !important;
            color: #CBD5E1 !important;
        }

        div[data-testid="stSidebar"] .stButton > button:hover,
        div[data-testid="stSidebar"] .stButton > button:focus {
            border-color: #DC2626 !important;
            color: #DC2626 !important;
            background: rgba(220, 38, 38, 0.1) !important;
        }

        div[data-testid="stCheckbox"] label,
        div[data-testid="stCheckbox"] label span,
        div[data-testid="stSidebar"] label[data-baseweb="checkbox"] {
            color: #CBD5E1 !important;
        }

        div[data-testid="stCheckbox"] label:hover,
        div[data-testid="stSidebar"] label[data-baseweb="checkbox"]:hover {
            color: #DC2626 !important;
        }

        div[data-testid="stCheckbox"] input:checked + div,
        div[data-testid="stSidebar"] input[type="checkbox"]:checked + div {
            background-color: #DC2626 !important;
            border-color: #DC2626 !important;
        }

        div[data-testid="stMetric"] {
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid var(--pmc-glass-border);
            border-radius: 12px;
            padding: 0.75rem 0.9rem;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.22);
        }

        div[data-testid="stMetric"] label {
            color: #94A3B8 !important;
        }

        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #FFFFFF !important;
        }

        /* ── Tabs: force RED active indicator ── */
        div[data-testid="stTabs"] {
            background: transparent;
        }

        div[data-testid="stTabs"] > div:first-child {
            background: rgba(30, 41, 59, 0.5);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 12px 12px 0 0;
            border: 1px solid var(--pmc-glass-border);
            border-bottom: none;
        }

        div[data-testid="stTabs"] [data-baseweb="tab-panel"] {
            background: rgba(30, 41, 59, 0.7) !important;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid var(--pmc-glass-border);
            border-top: none;
            border-radius: 0 0 12px 12px;
            padding: 1rem;
        }

        div[data-testid="stTabs"] button,
        .stTabs [data-baseweb="tab"] {
            border-radius: 12px 12px 0 0 !important;
            font-weight: 600 !important;
            color: #64748B !important;
            transition: color 0.2s ease, border-color 0.2s ease, background 0.2s ease !important;
        }

        div[data-testid="stTabs"] button:hover,
        .stTabs [data-baseweb="tab"]:hover {
            color: #DC2626 !important;
        }

        div[data-testid="stTabs"] button[aria-selected="true"],
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            color: #DC2626 !important;
            border-bottom: 2px solid #DC2626 !important;
            border-bottom-color: #DC2626 !important;
            background: rgba(30, 41, 59, 0.85) !important;
        }

        div[data-testid="stTabs"] [data-baseweb="tab-highlight"],
        .stTabs [data-baseweb="tab-highlight"],
        div[data-testid="stTabs"] [data-baseweb="tab-border"] {
            background-color: #DC2626 !important;
            background: #DC2626 !important;
        }

        /* ── Primary buttons: force RED ── */
        .stApp .stButton > button[kind="primary"],
        .stApp .stButton > button[data-testid="baseButton-primary"],
        .stApp div[data-testid="stDownloadButton"] > button[kind="primary"],
        .stApp div[data-testid="stDownloadButton"] > button[data-testid="baseButton-primary"],
        .stApp button[kind="primary"] {
            border-radius: 12px !important;
            font-weight: 700 !important;
            padding: 0.72rem 1.4rem !important;
            background: #DC2626 !important;
            background-color: #DC2626 !important;
            background-image: none !important;
            color: #FFFFFF !important;
            border: 1px solid #DC2626 !important;
            border-color: #DC2626 !important;
            box-shadow: 0 8px 20px rgba(220, 38, 38, 0.24) !important;
            transition: background 0.25s ease, background-color 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease, transform 0.2s ease !important;
        }

        .stApp .stButton > button[kind="primary"]:hover,
        .stApp .stButton > button[data-testid="baseButton-primary"]:hover,
        .stApp div[data-testid="stDownloadButton"] > button[kind="primary"]:hover,
        .stApp div[data-testid="stDownloadButton"] > button[data-testid="baseButton-primary"]:hover,
        .stApp button[kind="primary"]:hover {
            background: #B91C1C !important;
            background-color: #B91C1C !important;
            background-image: none !important;
            border-color: #B91C1C !important;
            color: #FFFFFF !important;
            transform: translateY(-1px);
            box-shadow: 0 10px 24px rgba(185, 28, 28, 0.3) !important;
        }

        .stApp .stButton > button[kind="primary"] p,
        .stApp .stButton > button[kind="primary"] span,
        .stApp .stButton > button[kind="primary"] div,
        .stApp .stButton > button[data-testid="baseButton-primary"] p,
        .stApp div[data-testid="stDownloadButton"] > button p,
        .stApp div[data-testid="stDownloadButton"] > button span {
            color: #FFFFFF !important;
        }

        div[data-testid="stStatusWidget"] {
            border-radius: 12px !important;
            border: 1px solid var(--pmc-glass-border) !important;
            background: rgba(30, 41, 59, 0.7) !important;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.22) !important;
            color: #CBD5E1 !important;
        }

        div[data-testid="stExpander"] {
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid var(--pmc-glass-border);
            border-radius: 12px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.22);
        }

        div[data-testid="stExpander"] summary,
        div[data-testid="stExpander"] summary span,
        div[data-testid="stExpander"] p {
            color: #CBD5E1 !important;
        }

        div[data-testid="stAlert"] {
            border-radius: 12px;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
        }

        div[data-testid="stProgress"] label {
            color: #CBD5E1 !important;
        }

        .stApp .stButton > button[kind="secondary"],
        .stApp .stButton > button[data-testid="baseButton-secondary"] {
            background: rgba(30, 41, 59, 0.7) !important;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid var(--pmc-glass-border) !important;
            color: #CBD5E1 !important;
            border-radius: 12px !important;
        }

        .stApp .stButton > button[kind="secondary"]:hover,
        .stApp .stButton > button[data-testid="baseButton-secondary"]:hover {
            border-color: #DC2626 !important;
            color: #FFFFFF !important;
            background: rgba(220, 38, 38, 0.12) !important;
        }

        div[data-testid="stCaptionContainer"] {
            color: #94A3B8 !important;
        }

        div[data-testid="stVerticalBlock"] > div[style*="border"] {
            background: rgba(30, 41, 59, 0.7) !important;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-color: var(--pmc-glass-border) !important;
            border-radius: 12px !important;
        }

        div[data-testid="stVideo"] {
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--pmc-glass-border);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
        }

        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }

        /* ── Login screen: dark glass on glowing background ── */
        .login-shell {
            max-width: 460px;
            margin: 3rem auto 0;
            padding: 1.75rem 1.75rem 0.25rem;
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid var(--pmc-glass-border);
            border-radius: 12px 12px 0 0;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }

        div[data-testid="column"]:has(.login-shell) div[data-testid="stForm"] {
            max-width: 460px;
            margin: 0 auto 2rem;
            padding: 0 1.75rem 1.5rem;
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid var(--pmc-glass-border);
            border-top: none;
            border-radius: 0 0 12px 12px;
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.35);
        }

        div[data-testid="column"]:has(.login-shell) label {
            color: #CBD5E1 !important;
        }

        .login-title {
            text-align: center;
            font-size: 1.55rem;
            font-weight: 800;
            color: #FFFFFF;
            margin-bottom: 0.35rem;
        }

        .login-subtitle {
            text-align: center;
            color: #94A3B8;
            margin-bottom: 1.5rem;
            line-height: 1.55;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session_state() -> None:
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "custom_notes" not in st.session_state:
        st.session_state.custom_notes = ""


def render_login_screen() -> None:
    st.markdown(
        """
        <div class="hero-wrap">
            <div class="hero-emoji">🔐</div>
            <h1 class="hero-title">
                <span class="hero-brand">PMC</span>
                <span class="hero-title-accent">| Prisijungimas</span>
            </h1>
            <div class="hero-title-sub">Vaizdo intelektas</div>
            <p class="hero-subtitle">
                Prašome prisijungti, kad pasiektumėte PMC vaizdo montažo platformą.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, login_col, _ = st.columns([1, 1.2, 1])
    with login_col:
        st.markdown(
            """
            <div class="login-shell">
                <div class="login-title">PMC | Prisijungimas</div>
                <div class="login-subtitle">Prašome prisijungti</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("pmc_login_form", clear_on_submit=False):
            username = st.text_input(
                "Vartotojo vardas",
                placeholder="Įveskite vartotojo vardą",
                label_visibility="visible",
            )
            password = st.text_input(
                "Slaptažodis",
                type="password",
                placeholder="Įveskite slaptažodį",
                label_visibility="visible",
            )
            submitted = st.form_submit_button("Prisijungti", type="primary", use_container_width=True)

            if submitted:
                if credentials_valid(username, password):
                    st.session_state["authenticated"] = True
                    st.rerun()
                else:
                    st.error("Neteisingas vartotojo vardas arba slaptažodis")


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero-wrap">
            <div class="hero-emoji">🎬</div>
            <h1 class="hero-title">
                <span class="hero-brand">PMC</span>
                <span class="hero-title-accent">| Atminties montažas</span>
            </h1>
            <div class="hero-title-sub">Vaizdo intelektas</div>
            <p class="hero-subtitle">
                PMC platforma trumpų vaizdo klipų analizei ir automatiniam geriausių momentų
                montažui. Pasirinkite kriterijus, įkelkite klipus, pridėkite fono muziką
                ir gaukite profesionalų vaizdo įrašą per kelias minutes.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> tuple[str, set[str], str]:
    with st.sidebar:
        st.markdown('<div class="pmc-sidebar-logo">PMC</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="pmc-sidebar-tagline">Vaizdo intelektas · Geriausių momentų montažas</div>',
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("### ⚙️ AI Modelio nustatymai")
        selected_model_label = st.selectbox(
            "Pasirinkite AI modelį:",
            options=list(GEMINI_MODEL_OPTIONS.keys()),
            index=list(GEMINI_MODEL_OPTIONS.keys()).index(DEFAULT_GEMINI_MODEL_LABEL),
            key="gemini_model_label",
        )
        selected_model_id = resolve_gemini_model(selected_model_label)
        st.caption(f"Naudojamas API modelis: `{selected_model_id}`")

        st.markdown("---")
        st.markdown('<span class="sidebar-badge">Kriterijai</span>', unsafe_allow_html=True)
        st.markdown("### 🎯 Momentų atranka")
        st.caption("Pažymėkite vieną ar kelias parinktis. AI sujungs jas į vieną užklausą.")

        selected_keys: list[str] = []
        for preset in CRITERIA_PRESETS:
            if st.checkbox(preset.label, key=f"criterion_{preset.key}"):
                selected_keys.append(preset.key)

        st.markdown("**Papildomi pastebėjimai**")
        custom_notes = st.text_area(
            "Papildomi pastebėjimai",
            height=88,
            placeholder="Pvz.: Parink momentus, kai vaikas šoka arba šoka šunelis...",
            label_visibility="collapsed",
            key="custom_notes",
        )

        criteria = build_criteria_descriptor(selected_keys, custom_notes)

        if criteria:
            st.markdown("**Suformuota užklausa Gemini:**")
            st.markdown(
                f'<div class="criteria-preview">{escape(criteria)}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("### ℹ️ Patarimai")
        st.info(
            "• Pažymėkite bent vieną parinktį arba įrašykite papildomus kriterijus\n\n"
            "• Kiekvienas klipas – iki **1 min.**\n\n"
            "• Formatai: **MP4, MOV, AVI**\n\n"
            "• Galite įkelti kelis klipus vienu metu"
        )

        api_ready = bool(os.getenv("GEMINI_API_KEY", "").strip())
        if api_ready:
            st.success("Gemini API raktas rastas ✓")
        else:
            st.warning("Trūksta GEMINI_API_KEY `.env` faile.")

        st.markdown("---")
        if st.button("Atsijungti", use_container_width=True, key="logout_button"):
            logout_user()

    return criteria, set(selected_keys), selected_model_id


def format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    minutes, secs = divmod(total, 60)
    if minutes:
        return f"{minutes} min. {secs} sek."
    return f"{secs} sek."


def render_upload_summary(uploaded_files: list[Any]) -> None:
    total_size_mb = sum(file.size for file in uploaded_files) / (1024 * 1024)
    formats = ", ".join(sorted({Path(file.name).suffix.lower().lstrip(".") for file in uploaded_files}))

    st.markdown(
        """
        <div class="panel-card">
            <div class="panel-title">📦 Įkelti klipai paruošti</div>
            <div class="panel-caption">Peržiūrėkite santrauką prieš generuojant vaizdo montažą.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_clips, col_size, col_formats = st.columns(3)
    with col_clips:
        st.metric("Paruošta klipų", len(uploaded_files))
    with col_size:
        st.metric("Bendras dydis", f"{total_size_mb:.1f} MB")
    with col_formats:
        st.metric("Formatai", formats.upper())

    with st.expander("Peržiūrėti failų sąrašą", expanded=False):
        for file in uploaded_files:
            st.write(f"🎞️ `{file.name}` · {file.size / (1024 * 1024):.1f} MB")


def render_results_section(
    output_path: Path,
    clips_processed: int,
    final_duration_sec: float,
    *,
    has_background_music: bool,
    music_mode_label: str,
) -> None:
    music_line = (
        f"Pridėta fono muzika ({music_mode_label})."
        if has_background_music
        else "Fono muzika nebuvo naudota."
    )
    st.markdown(
        f"""
        <div class="success-card">
            <h2 class="success-title">🎉 PMC vaizdo montažas paruoštas!</h2>
            <p class="success-text">
                AI atrinko geriausius momentus ir sujungė juos į vieną vaizdo įrašą.
                {escape(music_line)}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_left, metric_center, metric_right = st.columns(3)
    with metric_left:
        st.metric("Apdorota klipų", clips_processed)
    with metric_center:
        st.metric("Galutinė trukmė", format_duration(final_duration_sec))
    with metric_right:
        st.metric("Fono muzika", "Taip" if has_background_music else "Ne")

    st.markdown("#### 🍿 Peržiūra")
    _, video_col, _ = st.columns([1, 6, 1])
    with video_col:
        st.video(str(output_path))

        with open(output_path, "rb") as video_file:
            video_data = video_file.read()

        st.download_button(
            label="📥 Atsisiųsti galutinį video",
            data=video_data,
            file_name="PMC_Vaizdo_Montazas.mp4",
            mime="video/mp4",
            type="primary",
            use_container_width=True,
        )


def run_main_app() -> None:
    criteria, selected_criteria_keys, selected_model_id = render_sidebar()
    render_hero()

    uploaded_files: list[Any] | None = None
    background_music_file: Any | None = None
    logo_file: Any | None = None
    mix_with_original = True
    background_volume = DEFAULT_BACKGROUND_VOLUME

    with st.container():
        tab_settings, tab_upload = st.tabs(["1️⃣ Apžvalga", "2️⃣ Įkelti vaizdo įrašus"])

        with tab_settings:
            st.markdown(
                """
                <div class="panel-card">
                    <div class="panel-title">Kaip tai veikia</div>
                    <div class="panel-caption">
                        Trys paprasti žingsniai iki gražaus galutinio vaizdo įrašo.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            step_one, step_two, step_three = st.columns(3)
            with step_one:
                st.markdown("**1. Pasirinkite kriterijus**")
                st.write("Šoninėje juostoje pažymėkite parinktis ir, jei reikia, pridėkite pastabas.")
            with step_two:
                st.markdown("**2. Įkelkite klipus ir muziką**")
                st.write("Pasirinkite vaizdo failus (iki 1 min.), nuotraukas arba fono muziką.")
            with step_three:
                st.markdown("**3. Generuokite**")
                st.write("AI atrinks geriausius kadrus, sujungs juos ir pritaikys garso takelį.")

            if criteria.strip():
                st.success(f"**Dabartiniai kriterijai:** {criteria}")
                st.info(f"**Pasirinktas AI modelis:** `{selected_model_id}`")

        with tab_upload:
            st.markdown(
                """
                <div class="panel-card">
                    <div class="panel-title">Vaizdo failai</div>
                    <div class="panel-caption">
                        Įkelkite vaizdo įrašus arba nuotraukas. Leidžiami formatai: MP4, MOV, AVI, PNG, JPG.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            uploaded_files = st.file_uploader(
                "Įkelkite vaizdo failus arba nuotraukas",
                type=["mp4", "mov", "avi", "png", "jpg", "jpeg"],
                accept_multiple_files=True,
                help="Vilkite failus čia arba pasirinkite. Formatai: MP4, MOV, AVI, PNG, JPG, JPEG.",
            )

            if uploaded_files:
                render_upload_summary(uploaded_files)

            st.markdown(
                """
                <div class="panel-card">
                    <div class="panel-title">🎵 Fono muzika</div>
                    <div class="panel-caption">
                        Neprivaloma. Muzika bus pritaikyta tiksliai pagal galutinio vaizdo trukmę.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            background_music_file = st.file_uploader(
                "Pasirinkite fono muziką (neprivaloma)",
                type=["mp3", "wav"],
                accept_multiple_files=False,
                help="Leidžiami formatai: MP3, WAV.",
            )

            if background_music_file is not None:
                st.info(
                    f"Pasirinkta: `{background_music_file.name}` "
                    f"({background_music_file.size / (1024 * 1024):.1f} MB)"
                )
                mix_with_original = st.toggle(
                    "Sumaišyti su originaliu vaizdo garsu (išjungus – fono muzika pakeičia garsą)",
                    value=True,
                )
                background_volume = st.slider(
                    "Fono muzikos garsumas",
                    min_value=0.1,
                    max_value=1.0,
                    value=DEFAULT_BACKGROUND_VOLUME,
                    step=0.05,
                    help="Rekomenduojama 0.3, kad originalus garsas liktų girdimas.",
                )

            st.markdown(
                """
                <div class="panel-card">
                    <div class="panel-title">🏷️ Logotipas</div>
                    <div class="panel-caption">
                        Neprivaloma. PNG ar JPG logotipas bus rodomas viršutiniame dešiniajame kampe.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            logo_file = st.file_uploader(
                "Įkelkite logotipo paveikslėlį (PNG / neprivaloma)",
                type=["png", "jpg", "jpeg"],
                accept_multiple_files=False,
            )

            if logo_file is not None:
                st.info(
                    f"Pasirinktas logotipas: `{logo_file.name}` "
                    f"({logo_file.size / 1024:.1f} KB)"
                )

    st.markdown("")
    _, button_col, _ = st.columns([1, 2, 1])
    with button_col:
        generate_clicked = st.button(
            "🎬 Generuoti PMC vaizdo montažą",
            type="primary",
            use_container_width=True,
        )

    if not generate_clicked:
        return

    if not criteria.strip():
        st.error("Šoninėje juostoje pažymėkite bent vieną parinktį arba įrašykite papildomus kriterijus.")
        return

    if not uploaded_files:
        st.error("Skirtuke „Įkelti vaizdo įrašus“ įkelkite bent vieną failą.")
        return

    work_dir = Path(tempfile.mkdtemp(prefix="video_tvarkykle_"))
    progress_bar = st.progress(0, text="Pasiruošimas...")
    status_box = st.empty()

    try:
        saved_music_path: Path | None = None
        saved_logo_path: Path | None = None

        with st.spinner("📁 Ruošiami jūsų klipai..."):
            status_box.info("Išsaugomi įkelti failai į laikiną aplanką...")
            saved_paths = save_uploaded_media(uploaded_files, work_dir)
            if background_music_file is not None:
                saved_music_path = save_uploaded_audio(background_music_file, work_dir)
            if logo_file is not None:
                saved_logo_path = save_uploaded_logo(logo_file, work_dir)

        with st.status("PMC vaizdo intelektas apdoroja klipus...", expanded=True) as processing_status:
            processing_status.write("🤖 Gemini peržiūri jūsų klipus...")
            output_path = build_highlight_reel(
                media_items=saved_paths,
                criteria=criteria,
                model_id=selected_model_id,
                progress_bar=progress_bar,
                status_box=status_box,
                background_music_path=saved_music_path,
                mix_with_original=mix_with_original,
                background_volume=background_volume,
                prioritize_official_speeches=OFFICIAL_SPEECHES_KEY in selected_criteria_keys,
                logo_path=saved_logo_path,
            )
            if saved_logo_path is not None:
                processing_status.write("🏷️ Pridedamas logotipas ant galutinio vaizdo...")
            if saved_music_path is not None and OFFICIAL_SPEECHES_KEY in selected_criteria_keys:
                processing_status.write(
                    "🎵 Fono muzika pritaikyta – oficialių kalbų metu garsumas sumažintas..."
                )
            elif saved_music_path is not None:
                processing_status.write("🎵 Pridedama ir pritaikoma fono muzika...")
            processing_status.write("🎞️ Sujungiami geriausi momentai į vaizdo montažą...")
            processing_status.update(label="Vaizdo montažas sėkmingai sukurtas!", state="complete")

        with VideoFileClip(str(output_path)) as final_clip:
            final_duration = float(final_clip.duration or 0.0)

        music_mode_label = (
            "sumaišyta su originaliu garsu"
            if mix_with_original
            else "pakeičia originalų garsą"
        )
        render_results_section(
            output_path=output_path,
            clips_processed=len(saved_paths),
            final_duration_sec=final_duration,
            has_background_music=saved_music_path is not None,
            music_mode_label=music_mode_label,
        )
    except AppError as exc:
        progress_bar.empty()
        status_box.empty()
        st.error(str(exc))
        logger.warning("Naudotojo klaida: %s", exc)
    except Exception as exc:
        progress_bar.empty()
        status_box.empty()
        st.error("Įvyko netikėta klaida generuojant vaizdo įrašą. Patikrinkite serverio logus.")
        logger.exception("Netikėta klaida: %s", exc)
    finally:
        cleanup_directory(work_dir)


def main() -> None:
    configure_page()
    inject_custom_css()
    init_session_state()

    if not st.session_state.get("authenticated"):
        render_login_screen()
        return

    run_main_app()


if __name__ == "__main__":
    main()
