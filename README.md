# EV Adaptive Control under Input/State Constraints

Shared repo for the EV powertrain / BIBS project. Goal of this skeleton: a **working, mostly empty** pipeline
so each person can build their own segment, with automatic tests that tell us when something breaks.

## Layout

```
src/evcontrol/
  datagen/      references + bounded disturbances            (owners: TaehwanLee-KR, kaiyoshi08-dotcom)
  actuator/     u = sat(u_c), delta_u = u - u_c, EV limits   (owners: TaehwanLee-KR, kaiyoshi08-dotcom)
  evaluation/   RMSE, saturation fraction, UUB radius, ...   (owner: LiamJCurtis)
  pipeline.py   placeholder glue that runs all three together
tests/
  datagen/  actuator/  evaluation/   one folder of unit tests per segment
  integration/                       tests that cross segments
data/           data rules (big files are NOT stored in git)
docs/           GITHUB_SETUP.md  <- start here if you're new to GitHub
.github/        CI/CD pipeline, PR template, CODEOWNERS
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
make install                     # or: python -m pip install -e ".[dev]"
make test                        # run everything
make ci                          # lint + tests = what GitHub will run
```

## How the segments talk to each other (the "contract")

| Piece | Produces | Consumed by |
|-------|----------|-------------|
| `datagen` | 1-D numpy arrays `r(t)`, `d(t)` of length `n_steps` | pipeline / simulation |
| `actuator` | `(u, delta_u)` from a command `u_c` | pipeline / controller |
| `evaluation` | floats/ints from plain numpy arrays | scripts, reports |
| `pipeline.simulate` | dict with keys `t, r, d, x, u_c, u, delta_u` | evaluation |

Keep these shapes and key names stable. If you must change one, update `tests/integration/` in the same
pull request and tell the team. That is what the integration tests are for.

## The pipeline (what GitHub does for you)

| Workflow | Trigger | What it does |
|----------|---------|--------------|
| `ci.yml` | every push, every PR into `main` | lint -> unit tests per segment (py3.10 + 3.12) -> integration tests -> build package |
| `release.yml` | pushing a tag like `v0.2.0` | unit tests must pass (a skip or failure blocks the release unless the tag message contains `Override-Unit-Tests:`), then integration tests, build, and a GitHub Release |

## Adding your own work

1. Work only inside your segment folder and its matching `tests/<segment>/` folder.
2. Replace the `@pytest.mark.skip` placeholder tests with real ones as your code appears.
3. Open a pull request. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Roadmap (from the working notes)

Toy plant with one hard limit -> naive adaptive law -> sigma-mod / projection -> anti-windup state `z`
-> one barrier-function constraint -> simplified EV model -> reference governor.
