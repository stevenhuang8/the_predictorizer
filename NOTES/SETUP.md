# Setup log: tasks 1–3

Done on 2026-09-28, on an Apple Silicon Mac running macOS 26.3.1.

## Quick reference

```sh
docker compose up -d                   # start Postgres (container: eco-postgres)
docker compose down                    # stop it (data in ./postgres-data is kept)
docker compose --profile tools up -d   # also start pgAdmin at http://localhost:5050
set -a; . ./.env.local; set +a         # load the database settings into your shell
psql "$DATABASE_URL"                   # connect (note: port 5433, not 5432)
uv sync                                # install or refresh Python dependencies
uv run pytest / ruff check src / mypy src
```

---

## Task 1: install Docker (OrbStack)

**What I did**
1. `brew install --cask orbstack`
2. `open -a OrbStack`, then went through the first-run setup and chose **Docker** mode. This step has to be done in the app by hand.
3. Checked it with `docker --version` (29.4.0), `docker ps`, and `docker run --rm hello-world`.

**Things to remember**
- Before this, `/usr/local/bin/docker` was a link pointing into a Docker Desktop install that no longer existed. OrbStack's setup replaced it with a link to its own `docker`. The OrbStack command-line tools also live in `~/.orbstack/bin`.
- Choosing OrbStack over Docker Desktop: it's lighter and faster on Apple Silicon.

---

## Task 2: Python project (uv, `src/` layout)

**What I did**
1. `brew install uv` (0.12.19).
2. `uv init --lib --name eco-prediction --python 3.13 --vcs none .`
3. Set `requires-python = ">=3.11"` in `pyproject.toml`. The local interpreter is 3.13, set in `.python-version`.
4. Created the subpackages: `src/eco_prediction/{data,models,backtest,utils}/__init__.py`.
5. Installed dependencies:
   - `uv add pandas statsforecast lightgbm shap scikit-learn psycopg2-binary requests python-dotenv`
   - `uv add --dev pytest ruff mypy`
6. Added Python entries (`.venv/`, caches, `dist/`) to `.gitignore`.

**Problem: LightGBM wouldn't import**
- Error: `Library not loaded: @rpath/libomp.dylib`
- Cause: LightGBM needs the OpenMP runtime (`libomp`), which macOS doesn't include.
- Fix: `brew install libomp`. **This is needed on any new Mac you set the project up on.**

**Checks**
- `uv run python -c 'import pandas, lightgbm, shap, ...'` works.
- `ruff check src` and `mypy src` report no problems.

---

## Task 3: PostgreSQL 16 in Docker

**What I did**
1. Wrote `docker-compose.yml`:
   - Runs `postgres:16` in a container named `eco-postgres`.
   - Stores its data in `./postgres-data`, so it survives the container being removed.
   - Mounts `./migrations/init` as the setup-scripts folder. Those scripts **only run the first time**, when `postgres-data/` is empty.
   - Has a health check that uses `pg_isready`.
   - Includes an optional `pgadmin` service under the `tools` profile, on port 5050.
2. Added the database settings to `.env.local`: `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_HOST`, `POSTGRES_PORT` and `DATABASE_URL`.
   - The password was randomly generated with `openssl rand -hex 16`.
   - `.env.example` has the same settings with a `change_me` placeholder.
3. Added `postgres-data/` to `.gitignore`.

**Problem: port 5432 was already taken, so logins failed**
- Symptom: the container reported healthy, but `psql` from the Mac failed with `password authentication failed for user "eco"`, even though the password matched.
- Cause: a PostgreSQL **17** server from the EDB installer is installed natively (outside Docker):
  - It lives in `/Library/PostgreSQL/17`.
  - `/Library/LaunchDaemons/postgresql-17.plist` starts it at boot.
  - It owns `localhost:5432`, so connections were reaching it instead of the container.
  - `lsof` without sudo doesn't show it, because it runs as the system `postgres` user.
- Fix: exposed the container on **port 5433** by setting `POSTGRES_PORT=5433` and updating `DATABASE_URL`. I left the EDB server alone.
- To go back to 5432 later: change `POSTGRES_PORT` and `DATABASE_URL` in `.env.local`. The EDB server has since been disabled (see below).

**Follow-up: checked what uses PostgreSQL 17, then disabled it**
- What uses it: only the old `~/dev/stevebnb2` Django project (last commit June 2025). Its `settings.py` points at `localhost:5432`, and it probably uses the `stevebnb-1` database. Nothing else in `~/dev` uses it.
- Why disable it:
  - It was version 17.2 from late 2024, with no security updates since.
  - It listened on every network interface, and the macOS firewall allowed incoming connections to it, so other devices on the network could reach it.
  - Tools that assume port 5432 would quietly connect to it instead of the Docker database.
- How it was disabled (2026-09-28). These have to be run in a real Terminal, because the Claude Code `!` prefix has no terminal for sudo to ask for a password:
  ```sh
  sudo launchctl bootout system /Library/LaunchDaemons/postgresql-17.plist   # stop it now
  sudo launchctl disable system/postgresql-17                                 # don't start at boot
  ```
- Checked afterwards: `launchctl print-disabled system` shows `"postgresql-17" => disabled`, and nothing is listening on 5432.
- Its data is still in `/Library/PostgreSQL/17/data`. To use stevebnb again, run `sudo launchctl enable system/postgresql-17`, then `sudo launchctl bootstrap system /Library/LaunchDaemons/postgresql-17.plist`.
- To remove it completely: first back up with `pg_dump -U postgres -h localhost stevebnb-1 > ~/stevebnb-backup.sql` (while it's running), then run `/Library/PostgreSQL/17/uninstall-postgresql.app`.
- Loose end: `~/.psql_history` has a password in plain text from an old `ALTER USER` command. Change that password if you reuse it anywhere, and clear the file with `: > ~/.psql_history`.

**Checks**
- `psql "$DATABASE_URL"` connects and shows PostgreSQL 16.15 with an empty `eco_forecast` database.
- Connecting with psycopg2 plus `python-dotenv` from Python also works.
- To test that data persists, I created a table, ran `docker compose down` and `up`, and the row was still there. I dropped the table afterwards.

---

## What's next
- Task 4: the database schema and a migration system. Task 6: the FRED/ALFRED API client. Both are unblocked.
- Nothing is committed to git yet.
