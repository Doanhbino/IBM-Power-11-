"""Run identity, atomic records and hashed evidence; standard library only."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import uuid


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    """Replace one complete JSON record, including on failure/interruption."""
    path = Path(path)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.' + path.name, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def public_config(value):
    if isinstance(value, dict):
        return {key: '<redacted>' if key in {'source', 'password', 'token', 'secret'}
                else public_config(item) for key, item in value.items()}
    if isinstance(value, list):
        return [public_config(item) for item in value]
    if isinstance(value, str):
        return re.sub(r'\b(?:rtsp|rtsps|http|https)://[^\s]+', '<redacted-url>', value)
    return value


def identity(workload):
    return {'schema_version': 2, 'run_id': uuid.uuid4().hex, 'workload': workload,
            'created_utc': datetime.now(timezone.utc).isoformat()}


def artifact(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': sha256(path), 'bytes': path.stat().st_size}


def runtime_binaries(module):
    """Hash native runtime files, rather than treating a package version as a build ID."""
    root = Path(module.__file__).resolve().parent
    candidates = sorted(set(root.glob('capi/*.so*')) | set(root.glob('_C*.so')))
    # Training's loaded libtorch/backend libraries are separate from torch._C.
    if module.__name__ == 'torch':
        candidates += sorted(root.glob('lib/*.so*'))
    return [artifact(path) for path in candidates if path.is_file()]
