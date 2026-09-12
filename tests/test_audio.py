from pathlib import Path

import pytest

from meet_in_multi_language.audio import prepare_audio_chunks


def test_prepare_rejects_missing_source(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        prepare_audio_chunks(tmp_path / "missing.mp3", tmp_path / "output")


def test_prepare_rejects_invalid_segment_length(tmp_path: Path) -> None:
    source = tmp_path / "audio.mp3"
    source.write_bytes(b"audio")
    with pytest.raises(ValueError, match="片段秒數"):
        prepare_audio_chunks(source, tmp_path / "output", 0)
