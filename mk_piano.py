#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Dict

import pretty_midi
from yt_dlp import YoutubeDL


# ========= 基本設定 =========

PIANO_PROGRAM = 0  # Acoustic Grand Piano
DEFAULT_STEMS = ["vocals", "bass", "other"]  # drums は通常除外


# ========= 共通 =========

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


# ========= 1) ダウンロード =========

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


# ========= 2) Demucs で stem 分離 =========

def separate_stems(audio_path: Path, separate_root: Path, model: str = "htdemucs") -> Path:
    """
    Demucs 実行後、stem が入った曲ディレクトリを返す。
    出力例:
      separate_root/htdemucs/<track_name>/vocals.wav
    """
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

    # 一番新しい曲ディレクトリを採用
    subdirs = [p for p in model_dir.iterdir() if p.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"分離後の曲ディレクトリが見つかりません: {model_dir}")

    latest = sorted(subdirs, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    return latest


# ========= 3) stem -> MIDI =========

def transcribe_to_midi(audio_path: Path, midi_out_dir: Path) -> Path:
    """
    basic-pitch CLI:
      basic-pitch <output_dir> <input_audio>
    """
    midi_out_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["basic-pitch", str(midi_out_dir), str(audio_path)]
    run_command(cmd)

    mids = sorted(midi_out_dir.glob("*.mid"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not mids:
        raise FileNotFoundError(f"MIDI が生成されませんでした: {audio_path}")
    return mids[0]


# ========= 4) MIDI をピアノ化 =========

def normalize_midi_to_piano(midi_in: Path, midi_out: Path, velocity_scale: float = 1.0) -> Path:
    pm = pretty_midi.PrettyMIDI(str(midi_in))

    for inst in pm.instruments:
        if inst.is_drum:
            inst.notes.clear()
            continue
        inst.program = PIANO_PROGRAM
        for note in inst.notes:
            new_vel = int(max(1, min(127, round(note.velocity * velocity_scale))))
            note.velocity = new_vel

    midi_out.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(midi_out))
    return midi_out


# ========= 5) 複数 MIDI を重ねる =========

def merge_midis_to_single_song(
    stem_midi_map: Dict[str, Path],
    merged_midi_path: Path,
) -> Path:
    """
    stem ごとの MIDI を読み込み、1つの PrettyMIDI に全トラックを積む。
    各 stem は別トラックのまま残す。
    """
    merged = pretty_midi.PrettyMIDI()

    for stem_name, midi_path in stem_midi_map.items():
        pm = pretty_midi.PrettyMIDI(str(midi_path))

        if not pm.instruments:
            continue

        for i, inst in enumerate(pm.instruments):
            if inst.is_drum:
                continue

            new_inst = pretty_midi.Instrument(
                program=PIANO_PROGRAM,
                is_drum=False,
                name=f"{stem_name}_{i}"
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
        raise RuntimeError("結合対象の MIDI が空でした。")

    merged_midi_path.parent.mkdir(parents=True, exist_ok=True)
    merged.write(str(merged_midi_path))
    return merged_midi_path


# ========= 6) MIDI -> WAV =========

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


# ========= 曲単位処理 =========

def process_url(
    url: str,
    base_out_dir: Path,
    soundfont_path: Path,
    stems_to_use: List[str],
    demucs_model: str,
) -> None:
    print(f"\n=== Processing: {url} ===")

    raw_dir = base_out_dir / "01_downloaded_audio"
    sep_dir = base_out_dir / "02_separated"
    stem_midi_raw_dir = base_out_dir / "03_stem_midi_raw"
    stem_midi_piano_dir = base_out_dir / "04_stem_midi_piano"
    merged_dir = base_out_dir / "05_merged"
    wav_dir = base_out_dir / "06_wav"

    # 1. download
    audio_path = download_audio(url, raw_dir)
    song_name = sanitize_filename(audio_path.stem)

    # 2. separate
    song_sep_dir = separate_stems(audio_path, sep_dir, model=demucs_model)

    # 3. each stem -> MIDI -> piano MIDI
    merged_inputs: Dict[str, Path] = {}

    for stem_name in stems_to_use:
        stem_audio = song_sep_dir / f"{stem_name}.wav"
        if not stem_audio.exists():
            print(f"[SKIP] stem not found: {stem_audio}")
            continue

        print(f"[STEM] {stem_name}")

        raw_midi_dir = stem_midi_raw_dir / song_name / stem_name
        raw_midi_generated = transcribe_to_midi(stem_audio, raw_midi_dir)

        normalized_raw_midi = raw_midi_dir / f"{song_name}_{stem_name}.mid"
        if raw_midi_generated != normalized_raw_midi:
            raw_midi_generated.replace(normalized_raw_midi)
        raw_midi_generated = normalized_raw_midi

        piano_midi = stem_midi_piano_dir / song_name / f"{song_name}_{stem_name}_piano.mid"

        # stem ごとの音量補正
        velocity_scale = 1.0
        if stem_name == "vocals":
            velocity_scale = 1.15
        elif stem_name == "bass":
            velocity_scale = 0.95
        elif stem_name == "other":
            velocity_scale = 0.75

        normalize_midi_to_piano(raw_midi_generated, piano_midi, velocity_scale=velocity_scale)
        merged_inputs[stem_name] = piano_midi

    if not merged_inputs:
        raise RuntimeError("使える stem MIDI が1つも生成できませんでした。")

    # 4. merge as one song
    merged_midi = merged_dir / f"{song_name}_full_piano.mid"
    merge_midis_to_single_song(merged_inputs, merged_midi)

    # 5. render WAV
    merged_wav = wav_dir / f"{song_name}_full_piano.wav"
    render_midi_to_wav(merged_midi, merged_wav, soundfont_path)

    print(f"[OK] separated : {song_sep_dir}")
    print(f"[OK] merged mid: {merged_midi}")
    print(f"[OK] merged wav: {merged_wav}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="YouTube URL -> stem 分離 -> stemごとにMIDI化 -> ピアノ1曲へ統合"
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
        help="Demucs モデル名。通常は htdemucs のままでOK"
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
