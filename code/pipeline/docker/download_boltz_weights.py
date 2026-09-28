from pathlib import Path
from boltz.main import download_boltz2
download_boltz2(Path('/cache/boltz'))
print('Boltz-2 weights downloaded')
