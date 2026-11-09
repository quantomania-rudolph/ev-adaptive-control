#!/usr/bin/env bash
# Mac/Linux one-shot: new TimescaleDB container with a password you choose, then load all data.  UNTESTED (written without a Docker host).
set -euo pipefail
cd "$(dirname "$0")/.."
read -r -s -p "Choose a password for the new database (you will need it again): " PW; echo
echo "POSTGRES_PASSWORD=$PW" > db/.env
docker compose -f db/docker-compose.yml --env-file db/.env up -d
for i in $(seq 40); do docker compose -f db/docker-compose.yml --env-file db/.env exec -T tsdb pg_isready -U postgres && break; sleep 2; done
python -m pip install "psycopg[binary]" pandas pyarrow numpy
export PGURL="postgresql://postgres:${PW}@localhost:5432/powertrain"
python db/load_postgres.py --timescale
echo "Done. Connect with: psql \"$PGURL\""
