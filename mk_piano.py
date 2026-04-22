#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import pretty_midi
import soundfile as sf
from yt_dlp import YoutubeDL


PIANO_PROGRAM = 0  # Acoustic Grand Piano


# ==================================
# 共通ユーティリティ
# ==================================

def sanitize_filename(name: str, max_len: int = 140) -> str:
    name = re.sub(r'[\\/:*?"<>|]+', "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name[:max_len].rstrip(" .")


def read_links(txt_path: Path) -> List[str]:
    urls: List[str] = []
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line)
    return urls


def require_command(cmd_name: str) -> None:
    if shutil.which(cmd_name) is None:
        raise RuntimeError(f"'{cmd_name}' が見つかりません。PATH を確認してください。")


def run_command(cmd: List[str]) -> None:
    print(">", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"コマンド失敗: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


# ==================================
# 1) YouTube から音声ダウンロード
# ==================================

def download_audio(url: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "%(title)s [%(id)s].%(ext)s"),
        "noplaylist": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "0",
            }
        ],
        "quiet": False,
        "no_warnings": True,
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = sanitize_filename(info.get("title", "audio"))
        video_id = info.get("id", "unknown")
        wav_path = out_dir / f"{title} [{video_id}].wav"

    if wav_path.exists():
        return wav_path

    candidates = list(out_dir.glob(f"*[{video_id}].wav"))
    if len(candidates) == 1:
        return candidates[0]

    raise FileNotFoundError(f"WAV が見つかりません: {wav_path}")


# ==================================
# 2) Demucs で stem 分離
# ==================================

def separate_stems(
    audio_path: Path,
    separate_root: Path,
    model: str = "htdemucs",
) -> Path:
    separate_root.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "demucs",
        "-n", model,
        "-o", str(separate_root),
        str(audio_path),
    ]
    run_command(cmd)

    model_dir = separate_root / model
    if not model_dir.exists():
        raise FileNotFoundError(f"Demucs 出力ディレクトリが見つかりません: {model_dir}")

    subdirs = [p for p in model_dir.iterdir() if p.is_dir()]
    if len(subdirs) != 1:
        raise RuntimeError(
            f"Demucs の出力先が想定と異なります: {model_dir}\n"
            f"見つかった曲ディレクトリ: {[p.name for p in subdirs]}"
        )

    return subdirs[0]


# ==================================
# 3) bass / other は basic-pitch で MIDI 化
# ==================================

def transcribe_to_midi_basic_pitch(audio_path: Path, midi_out_dir: Path) -> Path:
    midi_out_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["basic-pitch", str(midi_out_dir), str(audio_path)]
    run_command(cmd)

    mids = sorted(midi_out_dir.glob("*.mid"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not mids:
        raise FileNotFoundError(f"MIDI が生成されませんでした: {audio_path}")

    return mids[0]


# ==================================
# 4) MIDI 共通後処理
# ==================================

def save_pretty_midi(pm: pretty_midi.PrettyMIDI, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(out_path))
    return out_path


def force_piano_inplace(pm: pretty_midi.PrettyMIDI) -> None:
    for inst in pm.instruments:
        inst.is_drum = False
        inst.program = PIANO_PROGRAM


def remove_short_notes_inplace(pm: pretty_midi.PrettyMIDI, min_duration_sec: float) -> None:
    for inst in pm.instruments:
        if inst.is_drum:
            inst.notes = []
            continue
        inst.notes = [note for note in inst.notes if (note.end - note.start) >= min_duration_sec]


def clamp_velocities_inplace(pm: pretty_midi.PrettyMIDI, velocity_scale: float = 1.0) -> None:
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            note.velocity = int(max(1, min(127, round(note.velocity * velocity_scale))))


def transpose_octave_inplace(pm: pretty_midi.PrettyMIDI, octave: int) -> None:
    shift = 12 * octave
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            note.pitch = max(0, min(127, note.pitch + shift))


def sort_notes_inplace(pm: pretty_midi.PrettyMIDI) -> None:
    for inst in pm.instruments:
        inst.notes.sort(key=lambda n: (n.start, n.pitch, n.end))


def prepare_stem_midi(
    midi_in: Path,
    midi_out: Path,
    stem_name: str,
    min_note_filter_sec: float,
    bass_octave_down: bool = False,
) -> Path:
    pm = pretty_midi.PrettyMIDI(str(midi_in))
    force_piano_inplace(pm)

    if stem_name == "bass":
        remove_short_notes_inplace(pm, min_note_filter_sec)
        clamp_velocities_inplace(pm, 0.95)
        if bass_octave_down:
            transpose_octave_inplace(pm, -1)
    elif stem_name == "other":
        remove_short_notes_inplace(pm, max(min_note_filter_sec, 0.06))
        clamp_velocities_inplace(pm, 0.68)
    else:
        remove_short_notes_inplace(pm, min_note_filter_sec)

    sort_notes_inplace(pm)
    return save_pretty_midi(pm, midi_out)


# ==================================
# 5) bass / other を 1 つの MIDI に結合
# ==================================

def merge_midis_to_single_song(
    stem_midi_map: Dict[str, Path],
    merged_midi_path: Path,
) -> Path:
    merged = pretty_midi.PrettyMIDI()

    for stem_name, midi_path in stem_midi_map.items():
        pm = pretty_midi.PrettyMIDI(str(midi_path))

        for idx, inst in enumerate(pm.instruments):
            if inst.is_drum:
                continue

            new_inst = pretty_midi.Instrument(
                program=PIANO_PROGRAM,
                is_drum=False,
                name=f"{stem_name}_{idx}",
            )

            for note in inst.notes:
                new_inst.notes.append(
                    pretty_midi.Note(
                        velocity=note.velocity,
                        pitch=note.pitch,
                        start=note.start,
                        end=note.end,
                    )
                )

            for cc in inst.control_changes:
                new_inst.control_changes.append(cc)
            for pb in inst.pitch_bends:
                new_inst.pitch_bends.append(pb)

            merged.instruments.append(new_inst)

    if not merged.instruments:
        raise RuntimeError("結合対象 MIDI が空でした。")

    return save_pretty_midi(merged, merged_midi_path)


# ==================================
# 6) MIDI -> WAV
# ==================================

def render_midi_to_wav(midi_path: Path, wav_path: Path, soundfont_path: Path) -> Path:
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "fluidsynth",
        "-ni",
        str(soundfont_path),
        str(midi_path),
        "-F",
        str(wav_path),
        "-r",
        "44100",
    ]
    run_command(cmd)

    if not wav_path.exists():
        raise FileNotFoundError(f"WAV の生成に失敗しました: {wav_path}")

    return wav_path


# ==================================
# 7) vocals: F0 推定
# ==================================

def median_filter_nan_safe(values: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return values.copy()

    if kernel_size % 2 == 0:
        kernel_size += 1

    pad = kernel_size // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    out = np.empty_like(values)

    for i in range(len(values)):
        window = padded[i:i + kernel_size]
        valid = window[~np.isnan(window)]
        out[i] = np.nan if len(valid) == 0 else np.median(valid)

    return out


def fill_small_unvoiced_gaps(
    f0_hz: np.ndarray,
    voiced_flag: np.ndarray,
    max_gap_frames: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    f0 = f0_hz.copy()
    voiced = voiced_flag.copy()

    n = len(f0)
    i = 0
    while i < n:
        if voiced[i]:
            i += 1
            continue

        j = i
        while j < n and not voiced[j]:
            j += 1

        gap_len = j - i
        left = i - 1
        right = j

        if (
            gap_len <= max_gap_frames
            and left >= 0
            and right < n
            and voiced[left]
            and voiced[right]
            and not np.isnan(f0[left])
            and not np.isnan(f0[right])
        ):
            interp = np.linspace(f0[left], f0[right], gap_len + 2)[1:-1]
            f0[i:j] = interp
            voiced[i:j] = True

        i = j

    return f0, voiced


def estimate_vocals_f0(
    vocals_wav: Path,
    sr: int = 22050,
    frame_length: int = 2048,
    hop_length: int = 256,
    fmin_hz: float = 110.0,
    fmax_hz: float = 1046.5,
    smooth_kernel: int = 7,
    max_unvoiced_gap_frames: int = 4,
) -> Tuple[np.ndarray, np.ndarray, int]:
    y, sr_loaded = librosa.load(str(vocals_wav), sr=sr, mono=True)

    f0_hz, voiced_flag, _ = librosa.pyin(
        y,
        sr=sr_loaded,
        fmin=fmin_hz,
        fmax=fmax_hz,
        frame_length=frame_length,
        hop_length=hop_length,
        switch_prob=0.01,
        no_trough_prob=0.01,
    )

    if f0_hz is None or voiced_flag is None:
        raise RuntimeError(f"pYIN による F0 推定に失敗しました: {vocals_wav}")

    f0_smooth = f0_hz.copy()
    voiced_values = np.where(voiced_flag, f0_smooth, np.nan)
    voiced_values = median_filter_nan_safe(voiced_values, kernel_size=smooth_kernel)

    for i in range(len(f0_smooth)):
        if voiced_flag[i] and np.isfinite(voiced_values[i]):
            f0_smooth[i] = voiced_values[i]

    f0_smooth, voiced_flag = fill_small_unvoiced_gaps(
        f0_smooth,
        voiced_flag,
        max_gap_frames=max_unvoiced_gap_frames,
    )

    return f0_smooth, voiced_flag, hop_length


# ==================================
# 8) vocals: 連続音で合成
# ==================================

def lowpass_onepole(x: np.ndarray, cutoff_hz: float, sr: int) -> np.ndarray:
    if cutoff_hz <= 0:
        return x.copy()
    a = math.exp(-2.0 * math.pi * cutoff_hz / sr)
    y = np.zeros_like(x)
    prev = 0.0
    for i, v in enumerate(x):
        prev = (1.0 - a) * v + a * prev
        y[i] = prev
    return y


def synthesize_continuous_vocals_from_f0(
    f0_hz: np.ndarray,
    voiced_flag: np.ndarray,
    sr: int,
    hop_length: int,
    timbre: str = "violinish",
    vibrato_preserve: float = 1.0,
    gain: float = 0.22,
) -> np.ndarray:
    """
    F0 をそのまま連続音として合成する。
    timbre:
      - sine
      - triangle
      - violinish
    """
    n_frames = len(f0_hz)
    n_samples = n_frames * hop_length

    # サンプル単位へ伸張
    f0_sample = np.repeat(f0_hz, hop_length)[:n_samples]
    voiced_sample = np.repeat(voiced_flag.astype(np.float32), hop_length)[:n_samples]

    # 短い揺れを少しだけ残す
    valid = np.isfinite(f0_sample) & (f0_sample > 0) & (voiced_sample > 0.5)
    if np.any(valid):
        f0_interp = f0_sample.copy()
        idx = np.arange(len(f0_interp))
        valid_idx = idx[valid]
        f0_interp[~valid] = np.interp(idx[~valid], valid_idx, f0_interp[valid])

        # 大きな流れ + 細かい揺れ の分離
        trend = lowpass_onepole(f0_interp, cutoff_hz=8.0, sr=sr)
        detail = f0_interp - trend
        f0_used = trend + detail * vibrato_preserve
    else:
        f0_used = np.zeros_like(f0_sample)

    # 無声音は 0
    f0_used[~valid] = 0.0

    # 位相積分
    phase = np.cumsum(2.0 * np.pi * f0_used / sr)

    # 基本波形
    if timbre == "sine":
        sig = np.sin(phase)
    elif timbre == "triangle":
        sig = (2.0 / np.pi) * np.arcsin(np.sin(phase))
    elif timbre == "violinish":
        # 軽い倍音つき。高調波は少なめにして耳障りを抑える
        sig = (
            0.78 * np.sin(phase)
            + 0.22 * np.sin(2.0 * phase + 0.15)
            + 0.08 * np.sin(3.0 * phase + 0.30)
        )
    else:
        raise ValueError(f"unknown timbre: {timbre}")

    # 振幅エンベロープ
    amp = np.where(valid, 1.0, 0.0).astype(np.float32)
    amp = lowpass_onepole(amp, cutoff_hz=20.0, sr=sr)  # attack/release を滑らかに
    amp = np.clip(amp, 0.0, 1.0)

    # 少し表情を付ける
    if np.any(valid):
        norm_f0 = np.clip((f0_used - 180.0) / 500.0, 0.0, 1.0)
        dynamic = 0.80 + 0.20 * norm_f0
    else:
        dynamic = np.ones_like(sig)

    out = sig * amp * dynamic * gain

    # DC/耳障り対策
    out = out - np.mean(out)
    out = lowpass_onepole(out, cutoff_hz=6000.0, sr=sr)
    return out.astype(np.float32)


def save_wav_mono(wav_path: Path, y: np.ndarray, sr: int) -> Path:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(wav_path), y, sr)
    return wav_path


# ==================================
# 9) オーディオをミックス
# ==================================

def load_audio_mono(path: Path, sr: int) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32)


def mix_audios(
    accompaniment_wav: Path,
    vocals_synth_wav: Path,
    out_wav: Path,
    sr: int = 44100,
    accompaniment_gain: float = 1.0,
    vocals_gain: float = 1.0,
    limiter_headroom: float = 0.98,
) -> Path:
    acc = load_audio_mono(accompaniment_wav, sr=sr) * accompaniment_gain
    voc = load_audio_mono(vocals_synth_wav, sr=sr) * vocals_gain

    n = max(len(acc), len(voc))
    acc_pad = np.zeros(n, dtype=np.float32)
    voc_pad = np.zeros(n, dtype=np.float32)
    acc_pad[:len(acc)] = acc
    voc_pad[:len(voc)] = voc

    mix = acc_pad + voc_pad

    peak = float(np.max(np.abs(mix))) if len(mix) else 0.0
    if peak > limiter_headroom and peak > 0:
        mix = mix * (limiter_headroom / peak)

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), mix, sr)
    return out_wav


# ==================================
# 10) 曲単位の処理
# ==================================

def process_url(
    url: str,
    base_out_dir: Path,
    soundfont_path: Path,
    stems_to_use: List[str],
    demucs_model: str,
    min_note_filter_sec: float,
    bass_octave_down: bool,
    vocal_timbre: str,
    vocal_gain: float,
) -> None:
    print(f"\n=== Processing: {url} ===")

    raw_dir = base_out_dir / "01_downloaded_audio"
    sep_root = base_out_dir / "02_separated"
    stem_midi_raw_dir = base_out_dir / "03_stem_midi_raw"
    stem_midi_fixed_dir = base_out_dir / "04_stem_midi_fixed"
    merged_dir = base_out_dir / "05_merged"
    wav_dir = base_out_dir / "06_wav"
    synth_vocal_dir = base_out_dir / "07_vocal_synth"
    final_mix_dir = base_out_dir / "08_final_mix"

    # 1. download
    audio_path = download_audio(url, raw_dir)
    song_name = sanitize_filename(audio_path.stem)

    # 2. separate
    song_sep_root = sep_root / song_name
    song_sep_dir = separate_stems(audio_path, song_sep_root, model=demucs_model)

    # 3. vocals は F0 → 連続音
    vocals_synth_wav: Optional[Path] = None
    if "vocals" in stems_to_use:
        vocals_wav = song_sep_dir / "vocals.wav"
        if vocals_wav.exists():
            print(f"[F0->SYNTH] vocals: {vocals_wav}")
            f0_hz, voiced_flag, hop_length = estimate_vocals_f0(vocals_wav)
            synth = synthesize_continuous_vocals_from_f0(
                f0_hz=f0_hz,
                voiced_flag=voiced_flag,
                sr=22050,
                hop_length=hop_length,
                timbre=vocal_timbre,
                vibrato_preserve=1.0,
                gain=vocal_gain,
            )
            vocals_synth_wav = synth_vocal_dir / f"{song_name}_vocals_{vocal_timbre}.wav"
            save_wav_mono(vocals_synth_wav, synth, 22050)
            print(f"[OK] vocal synth wav: {vocals_synth_wav}")
        else:
            print(f"[SKIP] vocals stem not found: {vocals_wav}")

    # 4. bass / other は MIDI 化
    raw_stem_midis: Dict[str, Path] = {}

    for stem_name in stems_to_use:
        if stem_name == "vocals":
            continue

        stem_audio = song_sep_dir / f"{stem_name}.wav"
        if not stem_audio.exists():
            print(f"[SKIP] stem not found: {stem_audio}")
            continue

        raw_midi_dir = stem_midi_raw_dir / song_name / stem_name
        raw_midi_dir.mkdir(parents=True, exist_ok=True)

        print(f"[BASIC-PITCH] {stem_name}: {stem_audio}")
        raw_midi_generated = transcribe_to_midi_basic_pitch(stem_audio, raw_midi_dir)

        normalized_raw_midi = raw_midi_dir / f"{song_name}_{stem_name}.mid"
        if raw_midi_generated != normalized_raw_midi:
            if normalized_raw_midi.exists():
                normalized_raw_midi.unlink()
            raw_midi_generated.replace(normalized_raw_midi)

        raw_stem_midis[stem_name] = normalized_raw_midi

    # 5. stem MIDI を整える
    fixed_midis: Dict[str, Path] = {}

    for stem_name, raw_midi_path in raw_stem_midis.items():
        fixed_midi_path = stem_midi_fixed_dir / song_name / f"{song_name}_{stem_name}_fixed.mid"

        prepare_stem_midi(
            midi_in=raw_midi_path,
            midi_out=fixed_midi_path,
            stem_name=stem_name,
            min_note_filter_sec=min_note_filter_sec,
            bass_octave_down=bass_octave_down,
        )

        fixed_midis[stem_name] = fixed_midi_path
        print(f"[FIXED] {stem_name}: {fixed_midi_path}")

    accompaniment_wav: Optional[Path] = None
    if fixed_midis:
        merged_midi = merged_dir / f"{song_name}_accompaniment_piano.mid"
        merge_midis_to_single_song(fixed_midis, merged_midi)

        accompaniment_wav = wav_dir / f"{song_name}_accompaniment_piano.wav"
        render_midi_to_wav(merged_midi, accompaniment_wav, soundfont_path)

        print(f"[OK] accompaniment mid: {merged_midi}")
        print(f"[OK] accompaniment wav: {accompaniment_wav}")

    # 6. ミックス
    final_mix_wav = final_mix_dir / f"{song_name}_final_mix.wav"

    if accompaniment_wav and vocals_synth_wav:
        mix_audios(
            accompaniment_wav=accompaniment_wav,
            vocals_synth_wav=vocals_synth_wav,
            out_wav=final_mix_wav,
            sr=44100,
            accompaniment_gain=1.0,
            vocals_gain=1.0,
        )
        print(f"[OK] final mix: {final_mix_wav}")
    elif accompaniment_wav:
        # 伴奏だけ
        shutil.copy2(accompaniment_wav, final_mix_wav)
        print(f"[OK] final mix (accompaniment only): {final_mix_wav}")
    elif vocals_synth_wav:
        y = load_audio_mono(vocals_synth_wav, sr=44100)
        sf.write(str(final_mix_wav), y, 44100)
        print(f"[OK] final mix (vocals only): {final_mix_wav}")
    else:
        raise RuntimeError("出力できる音源がありませんでした。")

    print(f"[OK] separated : {song_sep_dir}")


# ==================================
# CLI
# ==================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="YouTube -> stem分離 -> vocalsはF0連続音合成 / bass・otherはMIDIピアノ -> ミックス"
    )
    parser.add_argument("--links", required=True, type=Path, help="URL一覧 txt")
    parser.add_argument("--out", default=Path("output"), type=Path, help="出力先")
    parser.add_argument("--soundfont", required=True, type=Path, help=".sf2 ファイル")
    parser.add_argument(
        "--stems",
        default="vocals,bass,other",
        help="使う stem をカンマ区切りで指定。例: vocals,bass,other"
    )
    parser.add_argument(
        "--demucs-model",
        default="htdemucs",
        help="Demucs モデル名。通常は htdemucs"
    )
    parser.add_argument(
        "--min-note-filter",
        type=float,
        default=0.05,
        help="この長さ未満のノートを削除(秒)"
    )
    parser.add_argument(
        "--bass-octave-down",
        action="store_true",
        help="bass を 1 オクターブ下げる"
    )
    parser.add_argument(
        "--vocal-timbre",
        choices=["sine", "triangle", "violinish"],
        default="violinish",
        help="主旋律の合成音色"
    )
    parser.add_argument(
        "--vocal-gain",
        type=float,
        default=0.22,
        help="主旋律シンセの基礎音量"
    )

    args = parser.parse_args()

    if not args.links.exists():
        print(f"links file not found: {args.links}", file=sys.stderr)
        return 1
    if not args.soundfont.exists():
        print(f"soundfont not found: {args.soundfont}", file=sys.stderr)
        return 1

    require_command("ffmpeg")
    require_command("fluidsynth")
    require_command("basic-pitch")

    urls = read_links(args.links)
    if not urls:
        print("URL がありません。", file=sys.stderr)
        return 1

    stems_to_use = [s.strip() for s in args.stems.split(",") if s.strip()]
    failed: List[str] = []

    for url in urls:
        try:
            process_url(
                url=url,
                base_out_dir=args.out,
                soundfont_path=args.soundfont,
                stems_to_use=stems_to_use,
                demucs_model=args.demucs_model,
                min_note_filter_sec=args.min_note_filter,
                bass_octave_down=args.bass_octave_down,
                vocal_timbre=args.vocal_timbre,
                vocal_gain=args.vocal_gain,
            )
        except Exception as e:
            failed.append(url)
            print(f"[ERROR] {url}\n{e}", file=sys.stderr)

    print("\n=== Done ===")
    if failed:
        print("失敗した URL:")
        for u in failed:
            print(" -", u)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
