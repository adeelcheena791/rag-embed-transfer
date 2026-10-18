# Human audit of NQ retrieval misses (design/addendum_v0.4.md)

Annotator: the author only. No AI labels and no AI pre-screen.

1. Open `miss_audit_v04.xlsx`. Read the `Instructions` sheet.
2. Label **all 100 items in `Screen1`** first (A, A_passage, B). Do not look at `Screen2` until Screen1 is finished.
3. Then label `Screen2` (C, D).
4. Do **not** open `key.csv` or `sample.json` before both screens are done: they contain the automatic answer-string flag and the gold ids.
5. Save the workbook under the same name and commit it (or send it back). Counts with Wilson 95% intervals are then added to the paper.

Built by `manuscript_ipm/revision_v04/scripts/make_audit_sheet.py` (seed 20261017; 414 e5-base-v2 Hit@5 misses on the sub-corpus among the 2,255 NQ generation queries; 100 drawn without replacement).

## Easier option: labeling page
`miss_audit_page.html` (built by `manuscript_ipm/revision_v04/scripts/make_audit_page.py`) shows the same 100 items in the
same order, one at a time, with buttons for A–D and a search box that highlights only the words the annotator types (no
automatic answer flag, no gold passage). Screen 2 unlocks after all Screen 1 labels are done. Published as a private claude.ai
artifact; labels are saved in its database (collection `labels`) and can be downloaded as CSV.
