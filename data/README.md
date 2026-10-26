# Data

Git is built for code, not big data. Rules for this repo:

- `data/raw/` and `data/processed/` are **ignored by git** (see `.gitignore`).
- Anything small enough to regenerate should be produced by `evcontrol.datagen` with a fixed seed instead of being stored.
- Real/external datasets (drive cycles, battery data): put them on the shared team drive, and document in this file **where they live, where they came from, and the license**.
- If the team later needs versioned large data, look at DVC or Git LFS. Don't commit big files to plain git; they can never be fully removed from history.

## Datasets in use

| Name | Where to get it | Used by | Notes |
|------|-----------------|---------|-------|
| (none yet) | | | |
