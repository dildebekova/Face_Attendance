"""Generate unique local secrets using Python standard library."""
import base64
import os
import secrets
from pathlib import Path
p=Path(__file__).parent / ".env"
if p.exists():
    raise SystemExit(".env уже существует. Не перезаписываем ключ шифрования.")
p.write_text("DB_PASSWORD="+secrets.token_hex(24)+"\nADMIN_KEY="+secrets.token_urlsafe(32)+"\nEMBEDDING_KEY="+base64.urlsafe_b64encode(os.urandom(32)).decode()+"\nMATCH_THRESHOLD=0.55\nMATCH_MARGIN=0.08\nPAD_THRESHOLD=0.85\n")
try: p.chmod(0o600)
except OSError: pass
print("Создан .env. Ключ входа: ADMIN_KEY. Сохраните .env в безопасном месте.")
