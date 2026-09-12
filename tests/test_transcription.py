from pathlib import Path
from types import SimpleNamespace

from meet_in_multi_language.models import Engine
from meet_in_multi_language.transcription import OpenAITranscriber


class FakeTranscriptions:
    def __init__(self, response: object) -> None:
        self.response = response
        self.request: dict[str, object] | None = None

    def create(self, **request: object) -> object:
        self.request = request
        return self.response


def fake_client(response: object) -> tuple[object, FakeTranscriptions]:
    transcriptions = FakeTranscriptions(response)
    client = SimpleNamespace(audio=SimpleNamespace(transcriptions=transcriptions))
    return client, transcriptions


def test_standard_transcription_adds_multilingual_context(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"audio")
    client, calls = fake_client(
        SimpleNamespace(text="今仔日 review API。", languages=[SimpleNamespace(code="zh-tw")])
    )

    result = OpenAITranscriber("test", client=client).transcribe(
        audio, Engine.TRANSCRIBE, ["Codex"]
    )

    assert result.text == "今仔日 review API。"
    assert result.detected_languages == ["zh-tw"]
    assert calls.request is not None
    assert calls.request["languages"] == ["zh-tw", "en", "ja"]
    assert calls.request["keywords"] == ["Codex"]


def test_diarization_normalizes_segments_to_milliseconds(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"audio")
    client, calls = fake_client(
        SimpleNamespace(
            text="",
            segments=[SimpleNamespace(start=1.25, end=2.5, speaker="A", text=" 你好 ")],
        )
    )

    result = OpenAITranscriber("test", client=client).transcribe(
        audio, Engine.DIARIZE, []
    )

    assert result.text == "你好"
    assert result.segments[0].start_ms == 1250
    assert result.segments[0].end_ms == 2500
    assert result.segments[0].speaker == "A"
    assert calls.request is not None
    assert calls.request["chunking_strategy"] == "auto"
