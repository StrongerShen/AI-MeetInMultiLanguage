from pathlib import Path

import httpx

from meet_in_multi_language.speaches import (
    MIXED_LANGUAGE_PROMPT,
    PROFILES,
    SpeachesTranscriber,
    _segments_from_words,
    _segments_from_response,
    _quality_flags,
)
from meet_in_multi_language.models import TranscriptRevisionKind


def test_breeze_request_uses_hotwords_but_omits_mixed_prompt(tmp_path: Path) -> None:
    audio = tmp_path / "sample.mp3"
    audio.write_bytes(b"audio")
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.read().decode("utf-8")
        return httpx.Response(
            200,
            json={
                "text": "按呢 review API。",
                "language": "zh",
                "words": [
                    {"start": 0.0, "end": 0.4, "word": "按呢"},
                    {"start": 0.5, "end": 0.9, "word": " review"},
                    {"start": 1.0, "end": 1.2, "word": " API。"},
                ],
                    "segments": [
                        {"start": 0.0, "end": 1.2, "text": "按呢 review API。"}
                    ],
            },
        )

    client = httpx.Client(
        base_url="http://asr.test/v1", transport=httpx.MockTransport(handler)
    )
    result = SpeachesTranscriber(
        "http://asr.test/v1", PROFILES["breeze"], client=client
    ).transcribe(audio, "breeze", ["Codex"])

    body = str(seen["body"])
    assert "paulpengtw/faster-whisper-Breeze-ASR-26" in body
    assert MIXED_LANGUAGE_PROMPT not in body
    assert "hotwords" in body and "Codex" in body
    assert "timestamp_granularities" not in body
    assert result.provider == "speaches"
    assert result.revision_kind == TranscriptRevisionKind.RAW_ASR
    assert result.source_revision_id is None
    assert result.text == "按呢 review API。"
    assert result.segments[0].start_ms == 0
    assert result.segments[0].end_ms == 1200


def test_word_timestamps_split_on_long_gap() -> None:
    segments = _segments_from_words(
        [
            {"start": 0.0, "end": 0.2, "word": "前段"},
            {"start": 1.2, "end": 1.5, "word": "後段"},
        ]
    )

    assert [segment.text for segment in segments] == ["前段", "後段"]


def test_large_v3_profile_does_not_lock_language() -> None:
    assert PROFILES["large-v3"].language is None
    assert PROFILES["large-v3"].vad_filter is True
    assert PROFILES["large-v3"].prompt == MIXED_LANGUAGE_PROMPT


def test_sentence_blocks_mislabeled_as_words_fall_back_to_segments() -> None:
    segments = _segments_from_response(
        {
            "words": [
                {
                    "start": 0.0,
                    "end": 0.4,
                    "word": "這不是一個詞而是一整段很長很長且不應套用短時間範圍的辨識文字內容",
                }
            ],
            "segments": [
                {"start": 0.0, "end": 30.0, "text": "這是完整段落"},
            ],
        },
        word_timestamps=True,
    )

    assert len(segments) == 1
    assert segments[0].text == "這是完整段落"
    assert segments[0].end_ms == 30000


def test_breeze_short_end_timestamp_is_extended_to_next_block() -> None:
    segments = _segments_from_response(
        {
            "duration": 60.0,
            "segments": [
                {
                    "start": 0.0,
                    "end": 0.4,
                    "text": "這是一整段超過二十八個字而且不可能只在零點四秒內說完的會議辨識內容",
                },
                {"start": 30.0, "end": 30.5, "text": "最後一段同樣很長而且必須延伸到音訊結束才能合理供使用者回聽定位"},
            ],
        },
        word_timestamps=False,
    )

    assert [(segment.start_ms, segment.end_ms) for segment in segments] == [
        (0, 30000),
        (30000, 60000),
    ]


def test_repetition_and_no_speech_are_flagged_without_deleting_raw_text() -> None:
    text = "謝謝 謝謝 謝謝 謝謝 謝謝 謝謝"
    flags = _quality_flags(text, {"no_speech_prob": 0.8, "compression_ratio": 3.0})

    assert flags == [
        "suspected_repetition_hallucination",
        "high_no_speech_probability",
        "high_compression_ratio",
    ]
