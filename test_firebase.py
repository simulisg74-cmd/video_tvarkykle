"""Bandomasis Firestore prisijungimas ir duomenų įrašymas."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import firebase_admin
from firebase_admin import credentials, firestore

PROJECT_ROOT = Path(__file__).resolve().parent
FIREBASE_KEY_PATH = PROJECT_ROOT / "firebase-key.json"
COLLECTION_NAME = "analizuoti_video"


def get_firestore_client() -> firestore.Client:
    if not FIREBASE_KEY_PATH.exists():
        raise FileNotFoundError(
            f"Nerastas Firebase raktų failas: {FIREBASE_KEY_PATH}. "
            "Įdėkite firebase-key.json į projekto šaknį."
        )

    if not firebase_admin._apps:
        cred = credentials.Certificate(str(FIREBASE_KEY_PATH))
        firebase_admin.initialize_app(cred)

    return firestore.client()


def write_test_document(db: firestore.Client) -> str:
    test_payload = {
        "source": "test_firebase.py",
        "file_name": "test_clip.mp4",
        "status": "pending",
        "criteria": "Testiniai vaizdo analizės kriterijai",
        "notes": "Bandomasis įrašas iš test_firebase.py",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "test_run": True,
    }

    doc_ref = db.collection(COLLECTION_NAME).document()
    doc_ref.set(test_payload)
    return doc_ref.id


def main() -> int:
    try:
        db = get_firestore_client()
        doc_id = write_test_document(db)

        snapshot = db.collection(COLLECTION_NAME).document(doc_id).get()
        print("Prisijungta prie Firestore sėkmingai.")
        print(f"Kolekcija: {COLLECTION_NAME}")
        print(f"Sukurtas dokumento ID: {doc_id}")
        print(f"Įrašyti duomenys: {snapshot.to_dict()}")
        return 0
    except FileNotFoundError as exc:
        print(f"Klaida: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Firestore klaida: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
