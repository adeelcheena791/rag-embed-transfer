# Prompt for Codex in VS Code (copy everything below the line into Codex)

Open the `v05_kit` folder in VS Code first (File → Open Folder), then paste this into Codex.

---

This folder is a frozen research-experiment kit. Read `AGENTS.md` and `README_RUN.md` first and follow them exactly.

Your task: set up this Windows laptop and start the experiment, then help me monitor it.

1. Check the environment and install only what is missing: Python 3.10+, the pip packages `pyarrow` and `huggingface_hub`,
   and Ollama (it must be running). Then run `python v05.py check` and show me the output. Continue only if it shows
   `RESULT: ready` and `Kit file: unmodified`.
2. Start the full run **in a separate PowerShell window**, so it keeps running for days independently of this chat and of any
   command timeout:
   ```
   Start-Process powershell -ArgumentList '-NoExit','-ExecutionPolicy','Bypass','-File','.\run_all.ps1' -WorkingDirectory (Get-Location)
   ```
   `run_all.ps1` prepares the data, downloads missing models, runs Part A and Part B in the fixed order and creates the
   results zip after each part. Do not run the individual steps yourself in parallel with it.
3. When I ask for progress, run `python v05.py status` and read the last lines of `results\log.txt`, and tell me how far it is
   and roughly how many hours are left.
4. If the run window shows `STOPPED`, `STOP` or a traceback: read `results\log.txt`, fix it only if it is an environment
   problem (Ollama not running, missing package, disk full, network), then start `run_all.ps1` again the same way — it
   resumes. If it is not clearly an environment problem, do nothing else and tell me to send the screenshot and
   `results\log.txt` to the author.

Never modify `v05.py`, `run_all.ps1`, anything in `inputs/`, `KIT_SHA256.txt` or `VERSION`; never change model tags, plans,
settings or order; never delete or edit `results/` or `prompts/`; never pull a model again once its run has started; never
ask for or handle any API key. The optional Gemini step is not part of this task.
