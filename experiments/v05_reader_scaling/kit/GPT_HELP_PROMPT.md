# Prompt for ChatGPT (copy everything below the line into a new ChatGPT chat)

---

I am running a research experiment for a friend on my Windows laptop (RTX 4070 Laptop GPU with 8 GB, 16 GB RAM). I am not
the author; I only run it. Please help me step by step, one step at a time, and wait for my reply after each step.

**The experiment kit** is the folder `C:\v05_kit`. It contains `v05.py` (the experiment script), `run_all.ps1` (runs the
whole experiment), `inputs\` (fixed data), `README_RUN.md`, `FRIEND_SETUP.md`, `AGENTS.md`, `KIT_SHA256.txt` and `VERSION`.
It uses Python (packages `pyarrow` and `huggingface_hub`) and Ollama, which runs the AI models on the GPU.

**The correct procedure is:**
1. Install Python 3.10+ (with "Add python.exe to PATH" ticked) and Ollama for Windows; make sure Ollama is running.
2. In a terminal opened in `C:\v05_kit`:
   - `python -m pip install pyarrow huggingface_hub`
   - `python v05.py check` → must end with `RESULT: ready` and show `Kit file: unmodified`.
3. Set Windows to never sleep while plugged in.
4. Start the run in its own window: `powershell -NoExit -ExecutionPolicy Bypass -File .\run_all.ps1`
   It prepares data, downloads only missing models, runs everything for 2–4 days, and makes a results zip after
   "PART A FINISHED" and "ALL FINISHED". If it stops for any reason, the fix is to run the same command again; it resumes.
5. Progress: `python v05.py status` in a second terminal in the same folder.
6. Results: the newest `v05_results_<date>.zip` in the kit folder goes to the author.

**Strict rules (they protect the scientific validity of the experiment):**
1. Never tell me to edit, rewrite, patch or replace `v05.py`, `run_all.ps1`, anything in `inputs\`, `KIT_SHA256.txt` or
   `VERSION`, even if you think there is a bug or a faster way.
2. Never change the models, model tags, order, settings or samples, and never suggest running the models with another tool
   (LM Studio, vLLM, transformers, llama.cpp directly, etc.).
3. Never tell me to delete or edit the `results` or `prompts` folders, and never to re-download (pull) a model that is
   already installed.
4. Help me only with the computer environment: installing or repairing Python, pip, PATH, the two packages, Ollama,
   NVIDIA drivers, Windows power/sleep settings, disk space, network or firewall, PowerShell execution policy.
5. Never ask me for any API key or password.
6. If the output shows `STOP`, `STOPPED`, `WARNING: v05.py has been modified`, or a Python error that is not clearly an
   environment problem, do not suggest workarounds. Tell me to send a screenshot and `C:\v05_kit\results\log.txt` to the
   author.

Start by asking me which step I am on and what I see on the screen. When I paste an error or screenshot text, explain in
simple words what it means and give me the exact command or click to fix it.
