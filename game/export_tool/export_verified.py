"""
D5: export verified clips from Supabase into the training corpus layout.

    game (browser) -> Supabase Storage + Postgres row per clip
        -> reviewer app (verdicts)
        -> THIS TOOL: verified_good clips ->
             recordings/<child_code>/<phoneme>/<child_code>_<phoneme>_ep-<ipa>_<date>_<time>_<take>.wav
             + per-child child_meta.json (age band, region?, devices, sessions,
               holdout flag, consent record id)
        -> existing LOSO harness (evaluation/harness/) ingests with zero code
           changes (dataset.py scans recordings/<speaker>/<phoneme>/*.wav and
           skips loose files like child_meta.json).

Also handles deletion re-sync (PRD 6.3): children tombstoned in the DB get
their exported directory removed from recordings/ (and optionally their
storage objects purged with --purge-storage). Consent records are retained.

Verdict resolution (PRD 7):
  - golden clips are reviewer-reliability seeds, never exported
  - an adjudication verdict (Daniel) always wins
  - confusion-pair phonemes (dh, th, s, sh, m, n) need two independent
    ordinary verdicts that agree; disagreement waits for adjudication
  - other phonemes need one ordinary verdict (majority if several)
  - only a final label of verified_good is exported; wrong_sound and
    unusable clips are kept in Supabase, tagged, and excluded from training

Usage:
    poetry run python game/export_tool/export_verified.py [--dry-run] [--purge-storage]
                                                          [--recordings-dir recordings]

Credentials come from the environment or from game/.env (gitignored):
    SUPABASE_URL=https://xxx.supabase.co
    SUPABASE_SECRET_KEY=sb_secret_...   # bypasses RLS; keep out of git
    # (legacy projects: SUPABASE_SERVICE_ROLE_KEY with the service_role JWT also works)

The game app itself lives in the light-haven-sites repo (apps/); this tool
stays here because it writes into this repo's recordings/ and runs in the
Poetry environment.

Requires `requests` and `soundfile` (both already in the Poetry environment).
"""
import argparse
import io
import json
import os
import re
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import requests
import soundfile

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECORDINGS_DIR = REPO_ROOT / "recordings"
DOTENV_PATH = Path(__file__).resolve().parents[1] / ".env"  # game/.env, gitignored


def load_dotenv(path: Path = DOTENV_PATH) -> None:
    """Minimal KEY=value loader so credentials can live in game/.env.

    Real environment variables always win over file values.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

CONFUSION_PHONEMES = {"dh", "th", "s", "sh", "m", "n"}

# phoneme directory name -> ep- filename token.
# Mirrors game/src/lib/phonemes.ts and is verified against the existing
# corpus filenames, including the historical a_ɑ -> ep-æ quirk. Keep in sync.
PHONEME_IPA = {
    "I_aɪ": "aɪ", "a_æ": "æ", "a_ɑ": "æ", "ai_eɪ": "eɪ", "b": "b",
    "ch": "tʃ", "d": "d", "dh": "ð", "e": "ɛ", "ee": "iː", "f": "f",
    "g": "g", "h": "h", "i": "ɪ", "ir_ɝ": "ɝ", "j_dʒ": "dʒ", "k": "k",
    "l": "l", "m": "m", "n": "n", "ng": "ŋ", "oa": "oʊ", "oo_uw": "uwː",
    "oo_ʊ": "ʊː", "ow_aʊ": "aʊ", "oy_oɪ_ɔɪ": "oɪ", "p": "p", "s": "s",
    "sh": "ʃ", "su_ʒ": "ʒ", "t": "t", "th": "θ", "u": "ʌ", "v": "v",
    "w": "w", "y_j": "j", "z": "z",
}


class SupabaseClient:
    def __init__(self, url: str, service_key: str):
        self.url = url.rstrip("/")
        self.headers = {"apikey": service_key}
        # Legacy service_role keys are JWTs and are also sent as a Bearer
        # token. New-format sb_secret_ keys must NOT be: a non-JWT value in
        # Authorization is rejected at the database even when it matches the
        # apikey header (see supabase.com/docs/guides/api/api-keys).
        if not service_key.startswith("sb_"):
            self.headers["Authorization"] = f"Bearer {service_key}"

    def select_all(self, table: str, query: str = "select=*") -> List[dict]:
        """Fetch every row of a table, paging through PostgREST."""
        rows: List[dict] = []
        page_size = 1000
        offset = 0
        while True:
            resp = requests.get(
                f"{self.url}/rest/v1/{table}?{query}",
                headers={**self.headers, "Range": f"{offset}-{offset + page_size - 1}"},
                timeout=60,
            )
            resp.raise_for_status()
            batch = resp.json()
            rows.extend(batch)
            if len(batch) < page_size:
                return rows
            offset += page_size

    def download(self, bucket: str, path: str) -> bytes:
        resp = requests.get(
            f"{self.url}/storage/v1/object/{bucket}/{path}",
            headers=self.headers,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.content

    def delete_object(self, bucket: str, path: str) -> None:
        resp = requests.delete(
            f"{self.url}/storage/v1/object/{bucket}/{path}",
            headers=self.headers,
            timeout=60,
        )
        # 404 is fine on re-sync (already purged)
        if resp.status_code not in (200, 404):
            resp.raise_for_status()

    def patch(self, table: str, query: str, body: dict) -> None:
        resp = requests.patch(
            f"{self.url}/rest/v1/{table}?{query}",
            headers={**self.headers, "Content-Type": "application/json",
                     "Prefer": "return=minimal"},
            json=body,
            timeout=60,
        )
        resp.raise_for_status()


def final_label(clip: dict, verdicts: List[dict]) -> Optional[str]:
    """Resolve a clip's final label, or None if review is incomplete."""
    adjudications = [v for v in verdicts if v["is_adjudication"]]
    if adjudications:
        # latest adjudication wins
        return sorted(adjudications, key=lambda v: v["created_at"])[-1]["verdict"]

    ordinary = [v for v in verdicts if not v["is_adjudication"]]
    if not ordinary:
        return None

    if clip["phoneme"] in CONFUSION_PHONEMES:
        if len({v["reviewer_id"] for v in ordinary}) < 2:
            return None  # second independent review still pending
        labels = {v["verdict"] for v in ordinary}
        if len(labels) > 1:
            return None  # disagreement -> waits for adjudication
        return labels.pop()

    counts = Counter(v["verdict"] for v in ordinary)
    top = counts.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return None  # tie -> waits for adjudication
    return top[0][0]


def audio_problem(data: bytes) -> Optional[str]:
    """Structural gate before a clip enters recordings/: must parse as audio,
    be 16kHz mono, and contain actual frames. Content judgments (silence,
    wrong sound) belong to the reviewers; this only blocks files that would
    poison the corpus or crash the harness (the base corpus's 54 zero-byte
    wavs are the cautionary tale - dataset.py/sf.read fails hard on them).
    The game client uploads 16kHz mono 16-bit PCM WAV (apps/src/lib/wav.ts
    in light-haven-sites), so anything else here is a real defect upstream."""
    try:
        audio, sr = soundfile.read(io.BytesIO(data))
    except Exception as e:
        return f"unreadable ({e})"
    if sr != 16000:
        return f"sample rate {sr}Hz, expected 16000Hz"
    if audio.ndim != 1:
        return f"{audio.shape[1]} channels, expected mono"
    if len(audio) < 1600:  # < 0.1s: header-only / truncated upload
        return f"only {len(audio)} samples (<0.1s)"
    return None


def parse_timestamp(iso: str) -> datetime:
    """Postgres trims trailing zeros from fractional seconds (e.g.
    '.81194'), but Python 3.9's fromisoformat only accepts exactly 3 or 6
    fractional digits - normalize to 6 before parsing."""
    iso = iso.replace("Z", "+00:00")
    m = re.match(r"^(.*?)\.(\d+)([+-]\d{2}:\d{2})?$", iso)
    if m:
        frac = (m.group(2) + "000000")[:6]
        iso = f"{m.group(1)}.{frac}{m.group(3) or ''}"
    return datetime.fromisoformat(iso)


def export_filename(child_code: str, phoneme: str, recorded_at: str, take: int) -> str:
    # 2026-07-14T19:22:31.123456+00:00 -> 20260714_192231
    ts = parse_timestamp(recorded_at)
    stamp = ts.strftime("%Y%m%d_%H%M%S")
    ipa = PHONEME_IPA[phoneme]
    return f"{child_code}_{phoneme}_ep-{ipa}_{stamp}_{take}.wav"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1].strip())
    parser.add_argument("--recordings-dir", type=Path, default=DEFAULT_RECORDINGS_DIR)
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would happen; write nothing")
    parser.add_argument("--purge-storage", action="store_true",
                        help="also delete tombstoned children's clips from Supabase Storage")
    args = parser.parse_args()

    load_dotenv()
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SECRET_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        print("ERROR: set SUPABASE_URL and SUPABASE_SECRET_KEY "
              "(env or game/.env; legacy SUPABASE_SERVICE_ROLE_KEY also works)",
              file=sys.stderr)
        return 2

    sb = SupabaseClient(url, key)

    children = {c["id"]: c for c in sb.select_all("children")}
    clips = sb.select_all("clips", "select=*&is_golden=eq.false")
    verdicts_by_clip: Dict[str, List[dict]] = {}
    for v in sb.select_all("verdicts"):
        verdicts_by_clip.setdefault(v["clip_id"], []).append(v)
    sessions = sb.select_all("game_sessions")
    consents = sb.select_all("consent_records")

    recordings_dir: Path = args.recordings_dir

    # ---- deletion re-sync first (PRD 6.3) --------------------------------
    tombstoned = [c for c in children.values() if c["tombstoned_at"]]
    for child in tombstoned:
        child_dir = recordings_dir / child["child_code"]
        if child_dir.exists():
            print(f"[delete] removing exported copies for {child['child_code']} ({child_dir})")
            if not args.dry_run:
                shutil.rmtree(child_dir)
        if args.purge_storage:
            for clip in clips:
                if clip["child_id"] == child["id"]:
                    print(f"[delete] purging storage object {clip['storage_path']}")
                    if not args.dry_run:
                        sb.delete_object("clips", clip["storage_path"])

    # ---- export verified_good clips --------------------------------------
    stats = Counter()
    exported_children = set()
    for clip in clips:
        child = children.get(clip["child_id"])
        if child is None or child["tombstoned_at"]:
            stats["skipped_tombstoned"] += 1
            continue
        if clip["phoneme"] not in PHONEME_IPA:
            print(f"[warn] unknown phoneme label '{clip['phoneme']}' on clip {clip['id']}")
            stats["unknown_phoneme"] += 1
            continue

        label = final_label(clip, verdicts_by_clip.get(clip["id"], []))
        if label is None:
            stats["review_pending"] += 1
            continue
        if label != "verified_good":
            stats[f"excluded_{label}"] += 1
            continue

        child_code = child["child_code"]
        out_dir = recordings_dir / child_code / clip["phoneme"]
        out_path = out_dir / export_filename(
            child_code, clip["phoneme"], clip["recorded_at"], clip["take_number"]
        )
        if out_path.exists():
            stats["already_exported"] += 1
            exported_children.add(child["id"])
            continue

        print(f"[export] {out_path.relative_to(recordings_dir)}")
        if not args.dry_run:
            data = sb.download("clips", clip["storage_path"])
            problem = audio_problem(data)
            if problem:
                # Not ingested and exported_at stays unset, so the clip is
                # retried on the next run once the upstream defect is fixed.
                print(f"[warn] clip {clip['id']} ({clip['storage_path']}): "
                      f"{problem} - NOT ingested")
                stats["invalid_audio"] += 1
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            # temp file + os.replace: a crash mid-write must not leave a
            # truncated wav in the corpus (observed once on NTFS with a
            # plain write - see evaluation/harness/embeddings_cache.py)
            tmp_path = out_path.with_suffix(".wav.tmp")
            tmp_path.write_bytes(data)
            os.replace(tmp_path, out_path)
            sb.patch("clips", f"id=eq.{clip['id']}",
                     {"exported_at": datetime.utcnow().isoformat() + "Z"})
        stats["exported"] += 1
        exported_children.add(child["id"])

    # ---- per-child metadata JSON (PRD 8) ----------------------------------
    for child_id in exported_children:
        child = children[child_id]
        child_sessions = [s for s in sessions if s["child_id"] == child_id]
        child_consents = [c for c in consents if c["child_id"] == child_id]
        child_clips = [c for c in clips if c["child_id"] == child_id]
        devices = sorted({c["user_agent"] for c in child_clips if c.get("user_agent")})
        meta = {
            "child_code": child["child_code"],
            "age_band": child["age_band"],
            "region": child.get("region"),
            "holdout": child["is_holdout"],
            "session_dates": sorted({s["started_at"][:10] for s in child_sessions}),
            "devices": devices,
            "consent_record_ids": [c["id"] for c in child_consents],
            "exported_at": datetime.utcnow().isoformat() + "Z",
        }
        meta_path = recordings_dir / child["child_code"] / "child_meta.json"
        if not args.dry_run:
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
        print(f"[meta] {meta_path.relative_to(recordings_dir)} "
              f"(holdout={child['is_holdout']}, age_band={child['age_band']})")

    print("\n--- export summary ---")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    holdout_n = sum(1 for cid in exported_children if children[cid]["is_holdout"])
    print(f"  children with exported data: {len(exported_children)} "
          f"({holdout_n} holdout)")
    if args.dry_run:
        print("  (dry run: nothing was written)")
    elif stats["exported"]:
        print("\nNew clips ingested. The harness picks them up automatically "
              "(dataset.py scans recordings/); typical next steps:")
        print("  poetry run poe holdout-eval      # holdout-children accuracy")
        print("  poetry run poe learning-curve    # accuracy vs N children, per age band")
    return 0


if __name__ == "__main__":
    sys.exit(main())
