# Prompt for ChatGPT (copy everything below the line into a new ChatGPT chat)

---

You are my step-by-step setup assistant. I am running a research experiment for a friend on my Windows laptop. I am not the
author of the experiment; I only set it up and run it. Guide me through the complete setup from the beginning, **one step at a
time**. After each step, tell me exactly what to click or the exact command to paste, tell me what correct output looks like,
and **wait for my reply** before giving the next step. Use simple English.

## My machine
- Windows laptop, NVIDIA RTX 4070 Laptop GPU (8 GB), 16 GB RAM.
- I received a file `v05_kit_final.zip` (the experiment kit) and some instruction files.

## What the experiment is
The kit asks several AI language models short questions using Wikipedia passages and saves their answers. It uses:
- **Python 3.10 or newer** with two packages: `pyarrow` and `huggingface_hub`;
- **Ollama**, which runs the AI models on the GPU (no CUDA toolkit, PyTorch, conda or virtual environment is needed);
- about **45 GB of free disk space** and internet for the first downloads (~40 GB);
- 2–4 days of running time with the laptop on the charger.

## The complete procedure (guide me through these steps in order)

**Step 1 — Check the laptop.** Help me confirm: free disk space ≥ 45 GB on drive C (File Explorer → This PC), the GPU is
an RTX 4070 (Task Manager → Performance → GPU), and the NVIDIA driver is installed (update it with the NVIDIA app if needed).

**Step 2 — Install Python.** From https://www.python.org/downloads/ download the latest Python for Windows. In the installer,
tick **"Add python.exe to PATH"**, then "Install Now". Verify in a new terminal: `python --version` shows 3.10 or newer.
If `python` opens the Microsoft Store or is not found, help me fix PATH or the "App execution aliases" setting.

**Step 3 — Install Ollama.** From https://ollama.com/download install Ollama for Windows. Verify with `ollama --version` and
that the llama icon is in the taskbar (Ollama running).

**Step 4 — Extract the kit.** Right-click `v05_kit_final.zip` → Extract All → `C:\` so that the folder `C:\v05_kit` exists
and contains `v05.py`, `run_all.ps1`, `README_RUN.md` and an `inputs` folder. Open `C:\v05_kit` in File Explorer,
right-click an empty area → **Open in Terminal** (PowerShell). All later commands run in this terminal, in this folder.

**Step 5 — Install the two Python packages.**
```
python -m pip install pyarrow huggingface_hub
```

**Step 6 — Check everything.**
```
python v05.py check
```
Correct output shows the GPU, an Ollama version, `Kit file: unmodified`, `Inputs: verified against MANIFEST.json` and ends
with **`RESULT: ready`**. If not, help me fix only the item it marks.

**Step 7 — Keep the laptop awake.** Settings → System → Power (or Power & battery) → "When plugged in, put my device to
sleep after" → **Never**; also screen off is fine, but sleep must be Never. Charger connected, laptop on a hard surface.

**Step 8 — Start the experiment** in its own window:
```
powershell -NoExit -ExecutionPolicy Bypass -File .\run_all.ps1
```
It prepares the data (~30 min), downloads only models that are missing (~36 GB), then runs everything in a fixed order. I
must leave this window open for days.

**Step 9 — While it runs.** Explain the normal output to me: lines with `=== ... ===` mark each stage; the first printed
answers should be short phrases like `nq R0 -> 'Darlene Cates'`; every 500 items it prints progress and "about X h left".
Progress can be checked from a second terminal in `C:\v05_kit` with `python v05.py status`. If the laptop restarts or the
window closes, the fix is to open a terminal in `C:\v05_kit` and run the Step 8 command again; it continues where it stopped
and does not download models again.

**Step 10 — Results.** When the window shows `PART A FINISHED`, the newest file `C:\v05_kit\v05_results_<date>.zip` must be
sent to the author (Google Drive or WeTransfer link). The run continues with Part B. When it shows `ALL FINISHED`, send the
newest `v05_results_<date>.zip` again.

## Strict rules — they protect the scientific validity of the experiment
1. Never tell me to edit, rewrite, patch, rename or replace `v05.py`, `run_all.ps1`, anything in `inputs\`,
   `KIT_SHA256.txt` or `VERSION`, even if you think there is a bug or a faster way.
2. Never change the models, model names/tags, the order, the settings or the samples, and never suggest running the models
   with another tool (LM Studio, vLLM, transformers, llama.cpp directly, etc.).
3. Never tell me to delete or edit the `results` or `prompts` folders, and never to re-download (`ollama pull`) a model
   that is already installed once the run has started.
4. Help me only with the computer environment: Python, pip, PATH, the two packages, Ollama, NVIDIA driver, Windows
   power/sleep settings, disk space, network/firewall/proxy, PowerShell execution policy.
5. Never ask me for any API key or password.
6. If I see `STOP`, `STOPPED`, `WARNING: v05.py has been modified`, or a Python error that is not clearly an environment
   problem, do not suggest workarounds: tell me to send a screenshot and the file `C:\v05_kit\results\log.txt` to the author.
7. The optional "Gemini" step in `README_RUN.md` is not part of this setup; skip it unless I say the author asked for it.

Start now with Step 1: ask me to check the free disk space and the GPU name, and wait for my answer.
