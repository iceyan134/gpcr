"""Extract .conda packages from pkgs/ to conda environment.

openff-toolkit is a metapackage; real Python files live in openff-toolkit-base, etc.
Conda tarballs use site-packages/ as top-level (no lib/python prefix).
"""
import zipfile, os, tarfile, tempfile, zstd, glob, sys, shutil

site_pkgs = '/opt/conda/lib/python3.12/site-packages'
pkgs_dir = '/opt/conda/pkgs'

extracted = 0
for pkg_path in sorted(glob.glob(f'{pkgs_dir}/*.conda')):
    with tempfile.TemporaryDirectory() as tmp:
        try:
            with zipfile.ZipFile(pkg_path) as zf:
                zf.extractall(tmp)
        except Exception:
            continue

        for f in sorted(os.listdir(tmp)):
            if f.startswith('pkg-') and f.endswith('.tar.zst'):
                tar_path = os.path.join(tmp, f)
                with open(tar_path, 'rb') as fd:
                    data = zstd.decompress(fd.read())
                with tempfile.NamedTemporaryFile(suffix='.tar', delete=False) as ttf:
                    ttf.write(data)
                    tar_tmp = ttf.name
                try:
                    with tarfile.open(tar_tmp) as tf:
                        for member in tf.getmembers():
                            if member.name.startswith('site-packages/'):
                                rel = member.name[len('site-packages/'):]
                                if not rel:
                                    continue
                                target = os.path.join(site_pkgs, rel)
                                if member.isdir():
                                    os.makedirs(target, exist_ok=True)
                                elif member.isfile() or member.issym():
                                    os.makedirs(os.path.dirname(target), exist_ok=True)
                                    tf.extract(member, tmp)
                                    src = os.path.join(tmp, member.name)
                                    if os.path.exists(src):
                                        shutil.copy2(src, target)
                                extracted += 1
                finally:
                    os.unlink(tar_tmp)
                break

print(f'Extracted {extracted} files from conda packages')
