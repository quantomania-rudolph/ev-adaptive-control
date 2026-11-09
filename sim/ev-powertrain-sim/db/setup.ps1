# Windows one-shot: new TimescaleDB container with a password you choose, then load all data.  UNTESTED (written without a Docker host).
#   powershell -ExecutionPolicy Bypass -File db\setup.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$pw = Read-Host "Choose a password for the new database (you will need it again)"
"POSTGRES_PASSWORD=$pw" | Set-Content -Encoding ascii db\.env
docker compose -f db\docker-compose.yml --env-file db\.env up -d
Write-Host "waiting for the database..."
for ($i = 0; $i -lt 40; $i++) { docker compose -f db\docker-compose.yml --env-file db\.env exec -T tsdb pg_isready -U postgres 2>$null; if ($LASTEXITCODE -eq 0) { break }; Start-Sleep 2 }
python -m pip install "psycopg[binary]" pandas pyarrow numpy
$env:PGURL = "postgresql://postgres:$pw@localhost:5432/powertrain"
python db\load_postgres.py --timescale
Write-Host "Done. Connect with: psql `"$env:PGURL`""
