# Setup guide for running the experiment

Thank you for running this. The experiment asks several AI models short questions, using Wikipedia passages, and saves
their answers. It runs on your laptop's GPU for about 2–4 days. You only need to set it up and keep the laptop on.

You received: **`v05_kit_final.zip`** (the experiment) and **`CODEX_PROMPT.md`** (for Codex, optional).

## What you need
- Windows laptop with the RTX 4070 GPU, charger connected.
- **At least 45 GB free disk space** (check in File Explorer → This PC).
- Internet for the first few hours (downloads of about 40 GB).

## Part 1. Install three things (about 20 minutes)

1. **Python**
   - Go to https://www.python.org/downloads/ and download the latest Python for Windows.
   - Run the installer. On the first screen, **tick "Add python.exe to PATH"**, then click "Install Now".
2. **Ollama** (runs the AI models)
   - Go to https://ollama.com/download, download the Windows version and install it.
   - After installation a small llama icon appears in the taskbar. That means it is running.
3. **NVIDIA driver** — usually already installed on gaming laptops. If unsure, update it with the
   NVIDIA app / GeForce Experience.

## Part 2. Prepare the folder

1. Right-click `v05_kit_final.zip` → **Extract All** → choose `C:\` → this creates `C:\v05_kit`.
2. **Do not edit any file inside `C:\v05_kit`.** The experiment checks its own files.

## Part 3. Stop the laptop from sleeping

Settings → System → Power (or Power & battery) → "When plugged in, put my device to sleep after" → **Never**.
Keep the laptop on a hard surface with free airflow. Avoid games or heavy programs while it runs.

## Part 4. Start the experiment — choose ONE way

### Way A: with Codex in VS Code
1. Open VS Code → **File → Open Folder** → choose `C:\v05_kit`.
2. Open Codex, copy everything **below the line** in `CODEX_PROMPT.md`, paste it, and send.
3. Codex checks the setup and opens a separate PowerShell window where the experiment runs.
   **Leave that window open.**

### Way B: without Codex (also easy)
1. Open `C:\v05_kit` in File Explorer. Right-click an empty area → **Open in Terminal**.
2. Paste these commands one at a time and press Enter after each:
   ```
   python -m pip install pyarrow huggingface_hub
   python v05.py check
   ```
   The check must end with **`RESULT: ready`** and show **`Kit file: unmodified`**.
3. Start the experiment:
   ```
   powershell -NoExit -ExecutionPolicy Bypass -File .\run_all.ps1
   ```
   **Leave this window open.**

## Part 5. While it runs
- First it prepares data (about 30 minutes) and downloads the models (depends on your internet).
- Then it runs for days. Every 500 questions it prints a line with the progress and **"about X h left"**. This is normal.
- The first answers printed should be short phrases, for example `nq R0 -> 'Darlene Cates'`.
- If the laptop restarts or the window closes, **start the same command again**. It continues where it stopped and does
  not download the models again.
- To see progress, open a second terminal in `C:\v05_kit` and run: `python v05.py status`

## Part 6. Send the results
- When the window shows **`PART A FINISHED`**, look in `C:\v05_kit` for the newest file named
  `v05_results_<date>.zip` and send it to Adeel (Google Drive or WeTransfer link). The experiment continues with Part B.
- When it shows **`ALL FINISHED`**, send the newest `v05_results_<date>.zip` again.

## If something goes wrong
- The window shows **`STOPPED`** or **`STOP`**: take a screenshot, and send it with the file `C:\v05_kit\results\log.txt`
  to Adeel.
- `Ollama: NOT REACHABLE`: open Ollama from the Start menu, then start the same command again.
- For help with installation problems, you can use ChatGPT with the prompt in `GPT_HELP_PROMPT.md`.
- Please do not try to fix the experiment files yourself, and do not let any AI tool change them.

The optional Gemini step in `README_RUN.md` is not needed unless Adeel asks for it.
