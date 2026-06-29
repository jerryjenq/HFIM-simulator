# HFIM Three-Drug PK Simulator

This local tool simulates a first-pass hollow-fiber PK setup for three drugs:

- Fosfomycin q6h with central and extra compartment concentration profiles.
- Imipenem loading dose plus continuous infusion targeting 9 mg/L in the central compartment.
- Relebactam as 2/3 of the imipenem target concentration.

The current version compares two practical setups:

- `q24_replacement`: the extra compartment is filled with drug at t=0 and fully replaced every 24 h with the same volume and solved concentration.
- `overflow`: extra volume stays fixed by overflow, with drug loss tracked.

For `q24_replacement`, the simulator solves the fosfomycin central stock concentration and extra replacement concentration from target central Cavg/Css and Cmax. By default, Cavg/Css is 150 mg/L, equivalent to AUC0-24 3600 mg*h/L, and Cmax is 250 mg/L. If both targets cannot be reached with the current flow, q6h frequency, and infusion duration, the app reports the closest solution and a warning.

## Run

```bash
python3 -m unittest discover -s tests
python3 -m hfim_simulator.cli --scenario q24_replacement --duration-h 168
python3 -m hfim_simulator.cli --scenario overflow --duration-h 24
python3 -m streamlit run run_app.py
```

The CLI writes CSV/JSON outputs to `outputs/` and saved runs to `data/hfim-simulations.sqlite`.
The Streamlit UI requires installing `requirements.txt`.

## Local Secrets

Do not write API keys into Python files. For Gemini integration, create a local `.env` file from `.env.example` and put your key there:

```bash
cp .env.example .env
```

Then edit `.env` locally so it contains `GEMINI_API_KEY=...`. The `.gitignore` keeps `.env` out of version control.

The Streamlit app includes an HFIM Setup Assistant. If `GEMINI_API_KEY` is available and `google-genai` is installed, it can call Gemini. If Gemini is not configured, the assistant falls back to local HFIM rule checks so the app still works.

## SQLite Standard

Simulation rows use stable unique keys and conditional updates:

- `simulation_runs`: one row per simulator run.
- `concentration_timepoints`: unique by `run_id + time_min + drug`.
- `preparation_rows`: unique by `run_id + drug + component`.

Rows are inserted only when new and updated only when the stored values actually change. Identical rows are counted as unchanged and are not rewritten.
