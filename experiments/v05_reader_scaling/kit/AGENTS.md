# Instructions for AI coding agents (Codex, Copilot, ChatGPT, Claude, etc.)

This folder is a frozen research-experiment kit. The person using you only runs it; the procedure is in `README_RUN.md`.
Guide them through `README_RUN.md` one step at a time and help only with their computer environment.

Strict rules — these protect the scientific validity of the experiment:
1. **Never modify** `v05.py`, anything in `inputs/`, `KIT_SHA256.txt` or `VERSION`. Do not rewrite, refactor, "improve",
   reformat or patch them, even if you see a possible bug or a faster way. If you think the script itself must change,
   stop and tell me to contact the author.
2. **Never change the experiment**: not the model names or tags, not the order of runs, not the `--plan` values, not the
   decoding settings, not the samples, and do not run models with other tools (LM Studio, vLLM, transformers, etc.).
3. **Never delete or edit** anything in the `results` folder, and never delete `prompts` after `prepare` has finished.
4. Do not pull or update a model again after its run has started.
5. You may help with the environment only: installing or updating Python, pip packages (`pyarrow`, `huggingface_hub`),
   Ollama, NVIDIA drivers, Windows power settings, disk space, PATH problems, firewall or network problems.
6. Never ask for, display or store an API key. The Gemini key is set only with `$env:GEMINI_API_KEY="..."` in the terminal
   by me.
7. If a command prints `STOP`, `WARNING` or a Python traceback that is not clearly an environment problem, do not try
   workarounds. Tell me to send the terminal screenshot and `results\log.txt` to the author.
8. Interrupting a run is safe; the fix for an interruption is always to run the same command again.
