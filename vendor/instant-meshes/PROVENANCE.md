# Instant Meshes — vendored binary provenance

- **File:** `Instant Meshes.exe`
- **Source:** https://instant-meshes.s3.eu-central-1.amazonaws.com/Release/instant-meshes-windows.zip
  (official prebuilt Windows x64 release of github.com/wjakob/instant-meshes)
- **Downloaded:** 2026-07-12
- **SHA256:** `f1fe5b5f56d1002ae61dbd76a3fd646229fc1fe15322e7a3744f8ce442c76b2d`
- **Size:** 3,449,856 bytes
- **License:** BSD-3-Clause, © 2015 Wenzel Jakob, Daniele Panozzo, Marco Tarini,
  Olga Sorkine-Hornung — verbatim in `LICENSE.txt`. Binary redistribution is
  permitted; the copyright + license text are reproduced here and in the repo `NOTICE`.

## Usage

Optional quad-dominant retopology for the game-ready output mode (`retopo.py`).
Invoked headlessly:

```
"Instant Meshes.exe" <in.obj> -o <out.obj> -v <quads> -d -c 30
```

`-v`/`-f` count **quads**, so `retopo.py` passes ~half the triangle target. When
the binary is missing or errors, `retopo.py` falls back to a pymeshlab
isotropic-remesh + tri-to-quad path, so it is not a hard requirement.

## Refreshing

The upstream S3 artifact is auto-rebuilt from master, so its hash drifts over
time. The SHA256 above is the exact byte-image vendored here. On any refresh,
re-download, re-verify it runs headless, and re-record the hash + date.
