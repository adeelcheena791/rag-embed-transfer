# Phase 4 experiment: laptop par chalane ka tareeqa (Roman Urdu)

Yeh kit paper ke naye experiment ke liye hai: Qwen3 ke 4 sizes (1.7B, 4B, 8B, 14B) ko **wahi passages** de kar dekhna
ke bara reader retrieval gain ka zyada faida uthata hai ya nahi. Retrieval dobara nahi chalti; sirf reader chalta hai.

Laptop: RTX 4070 Laptop GPU (8 GB), 16 GB RAM, Windows. Disk: kam az kam **35 GB khaali** (models ~23 GB + corpus ~5 GB).

---

## Step 1: Zaroori cheezein install karo (sirf ek dafa)

1. **Python** (3.10 ya naya): https://www.python.org/downloads/ se install karo.
   Install karte waqt neeche **"Add python.exe to PATH"** par tick zaroor lagana.
2. **Ollama**: https://ollama.com/download se Windows wala install karo. Install ke baad Ollama khud chalta rehta hai
   (neeche taskbar mein llama ka icon).
3. Is kit ki zip ko kisi folder mein extract karo, jaise `C:\v05_kit`. Phir us folder mein khaali jagah par
   **Shift + Right click → "Open in Terminal"** (ya PowerShell). Saari commands isi window mein chalengi.
4. Python packages:
   ```
   python -m pip install pyarrow huggingface_hub
   ```
5. Check:
   ```
   python v05.py check
   ```
   Aakhir mein **"RESULT: ready"** aana chahiye, aur GPU mein RTX 4070 likha hona chahiye.

## Step 2: Laptop ko sone se roko
- Charger laga kar rakho.
- Settings → System → Power → "When plugged in, put my device to sleep after" → **Never**.
- Laptop ko hawa wali jagah rakho (neeche se band na ho); games ya bhari apps band rakho.

## Step 3: Data aur prompts tayyar karo (sirf ek dafa, ~20–40 minute, internet chahiye)
```
python v05.py prepare
```
Yeh do pinned Wikipedia corpus download kar ke check karta hai aur saare prompts bana deta hai.
Aakhir mein "prepare: done" aana chahiye. Agar **STOP** likha aaye to `results\log.txt` mujhe bhejo.
Is ke baad `work` folder delete kar sakte ho (disk bachane ke liye).

## Step 4: Models download karo (internet, ~23 GB)
```
ollama pull qwen3:1.7b-q4_K_M
ollama pull qwen3:4b-q4_K_M
ollama pull qwen3:8b-q4_K_M
ollama pull qwen3:14b-q4_K_M
ollama pull qwen2.5:7b-instruct-q4_K_M
```
**Experiment shuru hone ke baad kisi model ko dobara pull/update mat karna.**

## Step 5: Experiment chalao (isi tarteeb mein)
Har command ko chalne do. Beech mein band ho jaye (bijli, restart, Ctrl+C) to **wahi command dobara chalao**;
jo ho chuka hai woh skip ho jata hai.

```
python v05.py run qwen3:1.7b-q4_K_M --plan core
python v05.py run qwen3:4b-q4_K_M --plan core
python v05.py run qwen3:8b-q4_K_M --plan core
python v05.py run qwen2.5:7b-instruct-q4_K_M --plan bridge
python v05.py determinism
python v05.py run qwen3:14b-q4_K_M --plan 14b
```
- Pehle 5 jawab screen par dikhte hain (jaise `nq R0 -> 'Darlene Cates'`). Chhote, seedhe jawab hone chahiyein.
- Har 500 items ke baad speed aur **bacha hua waqt** (about X h left) dikhta hai.
- 14B slow hai; agar bohat slow nikla to script khud pehle se tay shuda chhota sample (first500) le leta hai.
- Andaza: 1.7B, 4B aur 8B mila kar takreeban 1 din; 14B aur takreeban 1 din. Asal waqt laptop par depend karta hai.
- Progress dekhne ke liye (doosri terminal window mein): `python v05.py status`

## Step 6 (optional): Gemini anchor
Yeh kisi bhi computer par chal sakta hai (GPU nahi chahiye), lekin Step 3 (prepare) wahan bhi hona chahiye
— ya isi laptop par saath saath chala lo.

1. https://aistudio.google.com se free API key banao. **Key kisi ko na bhejo, Claude ko bhi nahi.**
2. Key sirf is terminal window ke liye set karo (file mein save nahi hoti):
   ```
   $env:GEMINI_API_KEY="yahan-apni-key"
   ```
   (Command Prompt ho to: `set GEMINI_API_KEY=yahan-apni-key`)
3. Kaun se models hain:
   ```
   python v05.py gemini --list-models
   ```
   List mein se ek **Flash** model ka naam chuno (jaise `gemini-2.5-flash` agar maujood ho) aur mujhe bata do.
4. Chalao:
   ```
   python v05.py gemini --model <model-ka-naam>
   ```
   Free quota khatam ho jaye to script khud ruk jayega; **agle din wahi command** chalao.

## Step 7: Results bhejo
```
python v05.py pack
```
Yeh `v05_results_<tareekh>.zip` banata hai. Yeh **ek zip** mujhe bhej do (Google Drive / WeTransfer link bhi chalega).
Beech mein bhi bhej sakte ho (jaise 1.7B aur 8B ke baad) — main analysis shuru kar dunga.

## Masla aaye to
- `Ollama: NOT REACHABLE` → Ollama app start karo (Start menu → Ollama).
- `model ... is not pulled` → us model ka `ollama pull ...` chalao.
- `STOP: ... thinking text` → ruk jao, `results\log.txt` aur results file mujhe bhejo.
- Koi aur error → terminal ka screenshot aur `results\log.txt` bhejo.
