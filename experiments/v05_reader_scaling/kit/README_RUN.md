# Phase 4 experiment: how to run it on the laptop

**What this does.** The kit gives several language models (the "readers") exactly the same questions and the same five
Wikipedia passages that earlier readers saw in our study, and records their short answers. Retrieval is not rerun; only the
readers run. We then measure whether larger readers turn retrieval gains into more correct answers.

**Machine.** RTX 4070 Laptop GPU (8 GB), 16 GB RAM, Windows. Free disk: **at least 45 GB** (models about 36 GB, corpus
about 5 GB, results under 1 GB).

**The one rule.** Do not edit `v05.py` or anything in `inputs/`, and do not change any command or setting below. The script
checks itself and records any change. If something does not work, stop and send the error to the author (see the end).

---

## Step 1. Install the tools (once)

1. **Python 3.10 or newer** from https://www.python.org/downloads/. During installation, tick
   **"Add python.exe to PATH"**.
2. **Ollama for Windows** from https://ollama.com/download. After installation Ollama runs in the background
   (llama icon in the taskbar).
3. Extract this zip to a short folder path, for example `C:\v05_kit`. Open that folder, right-click on an empty area and
   choose **"Open in Terminal"** (PowerShell). Run every command below in this window, inside this folder.
4. Install the two Python packages:
   ```
   python -m pip install pyarrow huggingface_hub
   ```
5. Check the setup:
   ```
   python v05.py check
   ```
   The output must end with **`RESULT: ready`**, show **`Kit file: unmodified`**, and list the RTX 4070 under GPU.

## Step 2. Keep the laptop awake

- Keep the charger connected.
- Settings → System → Power → "When plugged in, put my device to sleep after" → **Never**.
- Put the laptop on a hard surface with free airflow. Close games and heavy programs while it runs.

**Shortcut.** After Step 2 you can run Steps 3–8 (without the optional Gemini step) with one command, in its own
PowerShell window opened in this folder:
```
powershell -NoExit -ExecutionPolicy Bypass -File .\run_all.ps1
```
It downloads only missing models, runs everything in the order below, and creates the results zip after Part A and at the
end. If it stops, start the same command again; it continues. Codex users: see `CODEX_PROMPT.md`.

## Step 3. Prepare the data (once, about 20–40 minutes, needs internet)
```
python v05.py prepare
```
This downloads two pinned Wikipedia corpora, checks them against the study's records and builds all prompts. It must end
with `prepare: done`. If it prints **STOP**, do not continue; send `results\log.txt` to the author.
Afterwards you may delete the `work` folder to free about 5 GB.

## Step 4. Download the models (needs internet, about 36 GB)
```
ollama pull qwen3:0.6b-q4_K_M
ollama pull qwen3:1.7b-q4_K_M
ollama pull qwen3:4b-q4_K_M
ollama pull qwen3:8b-q4_K_M
ollama pull qwen3:14b-q4_K_M
ollama pull qwen2.5:7b-instruct-q4_K_M
ollama pull gemma3:1b-it-q4_K_M
ollama pull gemma3:4b-it-q4_K_M
ollama pull gemma3:12b-it-q4_K_M
```
**Do not pull or update any of these models again once the runs have started.**

## Step 5. Run Part A (required), in this order
```
python v05.py run qwen3:0.6b-q4_K_M --plan core
python v05.py run qwen3:1.7b-q4_K_M --plan core
python v05.py run qwen3:4b-q4_K_M --plan core
python v05.py run qwen3:8b-q4_K_M --plan core
python v05.py run qwen2.5:7b-instruct-q4_K_M --plan bridge
python v05.py determinism
python v05.py run qwen3:14b-q4_K_M --plan 14b
```
- Each command can be stopped (power cut, restart, Ctrl+C) and **started again with the same command**; finished items are
  kept and skipped.
- The first five answers are printed, e.g. `nq R0 -> 'Darlene Cates'`. They should be short answers, not sentences.
- Every 500 items the script prints progress and the estimated time left.
- The 14B model is partly on the CPU and is slow. If it is too slow, the script switches by itself to the smaller sample
  fixed in the design. This is expected.
- Rough time: the four core runs together about one day; 14B about one more day. Your laptop decides the real time.
- To see progress from a second terminal window in the same folder: `python v05.py status`

**After Part A, send the results** (Step 8). The author can start the analysis while Part B runs.

## Step 6. Run Part B (second model family; run if time allows)
```
python v05.py run gemma3:1b-it-q4_K_M --plan core
python v05.py run gemma3:4b-it-q4_K_M --plan core
python v05.py run gemma3:12b-it-q4_K_M --plan large
```

## Step 7 (optional). Gemini comparison
This needs no GPU and can run on any computer where Step 3 has been done (or in a second window on the same laptop).

1. Create a free API key at https://aistudio.google.com. **Never share the key with anyone, including the author or any
   AI assistant, and never paste it into a file.**
2. Set the key for this terminal window only:
   ```
   $env:GEMINI_API_KEY="paste-your-key-here"
   ```
   (In Command Prompt instead of PowerShell: `set GEMINI_API_KEY=paste-your-key-here`)
3. List the available Flash models:
   ```
   python v05.py gemini --list-models
   ```
   Pick one Flash model id (for example `gemini-2.5-flash` if it is listed) and tell the author which one you used.
4. Run it:
   ```
   python v05.py gemini --model <model-id>
   ```
   When the free daily quota is used up, the script stops by itself. Run **the same command the next day**; it continues.

## Step 8. Send the results
```
python v05.py pack
```
This creates `v05_results_<date>.zip` in the kit folder. Send this one file to the author (Google Drive or WeTransfer link
is fine). You can send it after Part A and again at the end.

## If something goes wrong
| Message | What to do |
|---|---|
| `Ollama: NOT REACHABLE` | Start Ollama from the Start menu, then rerun the same command. |
| `model ... is not pulled` | Run the `ollama pull ...` line for that model, then rerun. |
| `STOP: ... thinking text` | Stop. Send `results\log.txt` and the results file to the author. |
| `WARNING: v05.py has been modified` | Extract the original zip again; do not edit the script. |
| Anything else | Take a screenshot of the terminal and send it with `results\log.txt` to the author. |
