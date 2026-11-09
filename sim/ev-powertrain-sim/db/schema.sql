-- Episode and training-run storage. UNTESTED against a live server (the build sandbox has no Postgres); db/pg_bench.py exercises it.
-- Learning note: metadata (what was run, with what settings, how it scored) is small and queried by many keys -> ordinary tables.
-- Per-step telemetry is huge and almost always read as "one run, maybe a time window" -> keyed by (run_id, k) and stored sorted.
CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS training_runs (          -- one row per training job / experiment campaign
  training_run_id bigserial PRIMARY KEY,
  name text NOT NULL, git_sha text, config jsonb NOT NULL DEFAULT '{}', started_at timestamptz DEFAULT now(), notes text);

CREATE TABLE IF NOT EXISTS runs (                   -- one row per episode (= one simulated run)
  run_id integer PRIMARY KEY,
  training_run_id bigint REFERENCES training_runs,
  family text NOT NULL,                              -- matrix | cycle | audit | tum_replay
  scenario text, controller text, seed integer, tau double precision,
  manifest jsonb NOT NULL DEFAULT '{}',              -- beliefs, sensor/inverter/BMS configs, fit path (the run's .json)
  metrics jsonb NOT NULL DEFAULT '{}',               -- per-run summary numbers (the *_metrics.csv row)
  rows integer, created_at timestamptz DEFAULT now());
CREATE INDEX IF NOT EXISTS runs_lookup ON runs (family, scenario, controller, seed);
CREATE INDEX IF NOT EXISTS runs_metrics_gin ON runs USING gin (metrics jsonb_path_ops);

CREATE TABLE IF NOT EXISTS catalog (                 -- the 184-column data dictionary, so units travel with the data
  name text PRIMARY KEY, unit text, grp text, role text, description text);

CREATE TABLE IF NOT EXISTS tum_samples (             -- TUM drive traces, one row per 10 Hz sample
  track text NOT NULL, t double precision NOT NULL, speed double precision, hv_battery_current double precision,
  hv_battery_voltage double precision, hv_soc double precision, hv_temp_min double precision, axle_torque double precision,
  rear_motor_torque double precision, accelerator_pedal double precision, elevation double precision, ambient_air_temp double precision,
  PRIMARY KEY (track, t));
-- steps (the wide telemetry table) is created by db/pg_bench.py in three layouts so they can be compared; pick one afterwards.
