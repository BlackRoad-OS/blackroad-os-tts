#!/usr/bin/env python3
"""
Text-to-speech engine manager for BlackRoad OS.
Supports multiple engines (pyttsx3, Google TTS) with fallback mocking.
"""

import os
import sqlite3
import json
import hashlib
import re
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import subprocess
import shutil

DB_PATH = Path.home() / ".blackroad" / "tts.db"
STORAGE_PATH = Path.home() / ".blackroad" / "storage" / "tts"


@dataclass
class Voice:
    """Represents a TTS voice."""
    id: str
    name: str
    language: str
    gender: str
    engine: str
    sample_rate: int
    description: str
    is_default: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Synthesis:
    """Represents a synthesis result."""
    id: str
    text: str
    voice_id: str
    speed: float
    pitch: float
    volume: float
    output_path: str
    duration_s: float
    created_at: str
    engine: str
    format: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TTSEngine:
    """Text-to-speech engine manager."""

    BUILT_IN_VOICES = [
        Voice("en-US-female-aria", "Aria", "en-US", "female", "pyttsx3", 24000,
              "American English female voice", is_default=True),
        Voice("en-US-male-atlas", "Atlas", "en-US", "male", "pyttsx3", 24000,
              "American English male voice"),
        Voice("en-GB-female-lucidia", "Lucidia", "en-GB", "female", "gtts", 22050,
              "British English female voice"),
        Voice("fr-FR-female-simone", "Simone", "fr-FR", "female", "gtts", 22050,
              "French female voice"),
        Voice("es-ES-male-carlos", "Carlos", "es-ES", "male", "gtts", 22050,
              "Spanish male voice"),
    ]

    def __init__(self):
        self._init_db()
        self._storage_path = STORAGE_PATH
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._pyttsx3_available = self._check_pyttsx3()
        self._gtts_available = self._check_gtts()

    def _init_db(self):
        """Initialize SQLite database."""
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS voices (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                language TEXT NOT NULL,
                gender TEXT,
                engine TEXT,
                sample_rate INTEGER,
                description TEXT,
                is_default BOOLEAN
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS syntheses (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                voice_id TEXT NOT NULL,
                speed REAL,
                pitch REAL,
                volume REAL,
                output_path TEXT,
                duration_s REAL,
                created_at TEXT,
                engine TEXT,
                format TEXT,
                FOREIGN KEY (voice_id) REFERENCES voices(id)
            )
        """)

        conn.commit()
        conn.close()

    def _check_pyttsx3(self) -> bool:
        """Check if pyttsx3 is available."""
        try:
            import pyttsx3
            return True
        except ImportError:
            return False

    def _check_gtts(self) -> bool:
        """Check if gTTS is available."""
        try:
            from gtts import gTTS
            return True
        except ImportError:
            return False

    def list_voices(self, language: Optional[str] = None, gender: Optional[str] = None) -> List[Voice]:
        """List available voices, optionally filtered by language or gender."""
        voices = self.BUILT_IN_VOICES.copy()

        if language:
            voices = [v for v in voices if v.language == language]
        if gender:
            voices = [v for v in voices if v.gender == gender]

        return voices

    def synthesize(self, text: str, voice_id: str, speed: float = 1.0, pitch: float = 1.0,
                   volume: float = 1.0, format: str = "mp3") -> Synthesis:
        """Synthesize text to speech."""
        voice = self._get_voice(voice_id)
        if not voice:
            raise ValueError(f"Voice {voice_id} not found")

        synthesis_id = hashlib.md5(f"{text}{voice_id}{datetime.now().isoformat()}".encode()).hexdigest()
        output_path = self._storage_path / f"{synthesis_id}.{format}"
        created_at = datetime.now().isoformat()

        # Estimate duration
        duration_s = self._estimate_duration(text, speed)

        # Try real synthesis
        if voice.engine == "pyttsx3" and self._pyttsx3_available:
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty('rate', 150 * speed)
                engine.setProperty('volume', volume)
                engine.save_to_file(text, str(output_path))
                engine.runAndWait()
            except Exception:
                self._create_mock_audio(output_path)
        elif voice.engine == "gtts" and self._gtts_available:
            try:
                from gtts import gTTS
                tts = gTTS(text=text, lang=voice.language.split('-')[0], slow=speed < 1.0)
                tts.save(str(output_path))
            except Exception:
                self._create_mock_audio(output_path)
        else:
            self._create_mock_audio(output_path)

        synthesis = Synthesis(
            id=synthesis_id,
            text=text,
            voice_id=voice_id,
            speed=speed,
            pitch=pitch,
            volume=volume,
            output_path=str(output_path),
            duration_s=duration_s,
            created_at=created_at,
            engine=voice.engine,
            format=format
        )

        self._save_synthesis(synthesis)
        return synthesis

    def batch_synthesize(self, texts: List[str], voice_id: str, output_dir: str) -> List[Synthesis]:
        """Synthesize multiple texts."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        results = []

        for i, text in enumerate(texts):
            synthesis = self.synthesize(text, voice_id, format="mp3")
            results.append(synthesis)

        return results

    def convert_format(self, input_path: str, output_format: str) -> str:
        """Convert audio format using ffmpeg if available."""
        input_file = Path(input_path)
        output_file = input_file.with_suffix(f".{output_format}")

        if shutil.which("ffmpeg"):
            try:
                subprocess.run(
                    ["ffmpeg", "-i", str(input_file), "-y", str(output_file)],
                    check=True, capture_output=True
                )
                return str(output_file)
            except subprocess.CalledProcessError:
                return str(input_file)
        else:
            # Fallback: just copy
            shutil.copy(input_file, output_file)
            return str(output_file)

    def get_synthesis(self, synthesis_id: str) -> Optional[Synthesis]:
        """Get synthesis by ID."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM syntheses WHERE id = ?", (synthesis_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return Synthesis(*row)
        return None

    def list_syntheses(self, voice_id: Optional[str] = None, language: Optional[str] = None) -> List[Synthesis]:
        """List recent syntheses, optionally filtered."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        query = "SELECT * FROM syntheses WHERE 1=1"
        params = []

        if voice_id:
            query += " AND voice_id = ?"
            params.append(voice_id)

        query += " ORDER BY created_at DESC LIMIT 100"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [Synthesis(*row) for row in rows]

    def get_stats(self) -> Dict[str, Any]:
        """Get TTS statistics."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("SELECT SUM(LENGTH(text)), AVG(duration_s), voice_id, engine FROM syntheses GROUP BY voice_id, engine")
        rows = cursor.fetchall()
        conn.close()

        total_chars = sum(row[0] or 0 for row in rows)
        avg_duration = sum(row[1] or 0 for row in rows) / len(rows) if rows else 0
        most_used = max(rows, key=lambda x: x[0] or 0) if rows else None

        return {
            "total_chars_synthesized": total_chars,
            "avg_duration_s": avg_duration,
            "most_used_voice": most_used[2] if most_used else None,
            "engine_breakdown": {row[3]: row[0] or 0 for row in rows}
        }

    def ssml_to_text(self, ssml: str) -> str:
        """Strip SSML tags and extract plain text."""
        # Simple regex-based SSML stripping
        text = re.sub(r'<[^>]+>', '', ssml)
        text = re.sub(r'&lt;', '<')
        text = re.sub(r'&gt;', '>')
        text = re.sub(r'&amp;', '&')
        return text.strip()

    def estimate_duration(self, text: str, speed: float = 1.0) -> float:
        """Estimate duration based on word count and speed."""
        return self._estimate_duration(text, speed)

    def _estimate_duration(self, text: str, speed: float) -> float:
        """Internal duration estimation."""
        words = len(text.split())
        avg_wps = 150  # Average words per second
        return (words / avg_wps) / speed

    def _get_voice(self, voice_id: str) -> Optional[Voice]:
        """Get voice by ID."""
        for voice in self.BUILT_IN_VOICES:
            if voice.id == voice_id:
                return voice
        return None

    def _save_synthesis(self, synthesis: Synthesis):
        """Save synthesis to database."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO syntheses
            (id, text, voice_id, speed, pitch, volume, output_path, duration_s, created_at, engine, format)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (synthesis.id, synthesis.text, synthesis.voice_id, synthesis.speed, synthesis.pitch,
              synthesis.volume, synthesis.output_path, synthesis.duration_s, synthesis.created_at,
              synthesis.engine, synthesis.format))

        conn.commit()
        conn.close()

    def _create_mock_audio(self, output_path: Path):
        """Create a mock audio file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Create a simple MP3/WAV mock file
        output_path.write_bytes(b'MOCK_AUDIO_DATA')


def main():
    """CLI interface."""
    import sys

    engine = TTSEngine()

    if len(sys.argv) < 2:
        print("Usage: python tts_engine.py [voices|speak|stats]")
        return

    cmd = sys.argv[1]

    if cmd == "voices":
        language = None
        if len(sys.argv) > 3 and sys.argv[2] == "--language":
            language = sys.argv[3]

        voices = engine.list_voices(language=language)
        for voice in voices:
            print(f"  {voice.id}: {voice.name} ({voice.language}, {voice.gender})")

    elif cmd == "speak":
        if len(sys.argv) < 3:
            print("Usage: python tts_engine.py speak '<text>' --voice <voice_id>")
            return

        text = sys.argv[2]
        voice_id = "en-US-female-aria"

        if len(sys.argv) > 4 and sys.argv[3] == "--voice":
            voice_id = sys.argv[4]

        synthesis = engine.synthesize(text, voice_id)
        print(f"Synthesized: {synthesis.id}")
        print(f"Output: {synthesis.output_path}")
        print(f"Duration: {synthesis.duration_s:.2f}s")

    elif cmd == "stats":
        stats = engine.get_stats()
        print(json.dumps(stats, indent=2))

    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
