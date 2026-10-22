# The vault

This disk is the **persistent storage volume**. It survives when the rented GPU
server is deleted. The server's own SSD does not. It is rewritten from
`remote/server/VAULT.md` in the vibration_anomaly_detection repo on every
`box up`. The full runbook is `remote/README.md` in that repo.

```
VAULT.md            this file
.vault-id           uuid + creation date (proves this is the same vault)
.mount-history      one line per mount: when, which host
env/                bin/uv, uv-cache/, apt-debs/  (fast installs on a fresh box)
datasets/<name>/<version>/
    MANIFEST.json   source, per-file size + sha256, status (complete = usable)
    SOURCE.md       license, citation, notes
    raw/            publisher's files (small datasets)
    condensed/      derived stores (Paderborn v1/v2, XJTU-SY .npy)
    SEALED          present = do not open before WP11 (IMS)
projects/<project>/
    outputs/        mirror of the repo's outputs/ (caches survive rentals)
    runs/<id>/      cmd.txt git.json diff.patch env.txt log exit_code duration_s
```

Rules:
- Mount it on **one server at a time** (ext4 is not a shared filesystem).
- Work on the SSD (`~/work`), not here. `syncd` copies results here every 60 s.
- Run `box down` before the rental ends: it syncs, pulls, and unmounts cleanly.
- Datasets are untrusted. Never run code from inside `datasets/`.
