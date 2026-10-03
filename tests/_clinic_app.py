"""A real clinic API (its own SQLite file, seeded with the sample patients) for tests that need one.

    with clinic_app() as (app, client):       # client is a starlette TestClient; app is the ASGI app

The clinic-api modules (main, models, registry, ...) share names with nothing else, except `main`, which
is also the orchestrator: this loads clinic-api's copy, keeps the app object, and puts sys.path and
sys.modules back afterwards, so it can be used next to tests/_pod_stubs.py.
"""

import contextlib
import os
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CLINIC_MODULES = (
    "main",
    "db",
    "models",
    "seed",
    "booking_service",
    "booking_migrate",
    "enquiry_service",
    "enquiry_migrate",
    "i18n_content",
    "phonetic_match",
    "patient_context",
    "disclosure",
    "registry",
    "agent_messages",
    "patient_seed",
    "gazetteer",
    "idempotency",
)


@contextlib.contextmanager
def clinic_app(sample_patients: bool = True):
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    saved_env = {k: os.environ.get(k) for k in ("CLINIC_DB_PATH", "DATABASE_URL", "CLINIC_SEED_SAMPLE_PATIENTS")}
    os.environ["CLINIC_DB_PATH"] = db_path
    os.environ.pop("DATABASE_URL", None)
    os.environ["CLINIC_SEED_SAMPLE_PATIENTS"] = "1" if sample_patients else "0"
    saved_path = list(sys.path)
    saved_mods = {k: sys.modules.pop(k) for k in _CLINIC_MODULES if k in sys.modules}
    sys.path.insert(0, os.path.join(REPO_ROOT, "clinic-api"))
    try:
        from fastapi.testclient import TestClient

        import main as clinic_main

        with TestClient(clinic_main.app) as client:
            yield clinic_main.app, client
    finally:
        for k in _CLINIC_MODULES:
            sys.modules.pop(k, None)
        sys.modules.update(saved_mods)
        sys.path[:] = saved_path
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        try:
            os.remove(db_path)
        except OSError:
            pass
