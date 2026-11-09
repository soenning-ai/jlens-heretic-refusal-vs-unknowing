# Output files

Raw results exactly as the measurement scripts wrote them. The scripts were written in
German, so the JSON field names are German. They are kept unchanged: the files are the
record, and renaming keys would make them something else.

## Layout

```
qwen35_{sft,heretic,basis}/facts_jacobian_lens_upper_20prompt.json            fact battery + lens per answer token
qwen35_{sft,heretic,basis}/refusal_signals_jacobian_lens_upper_20prompt.json  refusal pass, numbers only
gemma4_12b/facts_jacobian_lens_upper.json                                      fact battery, Gemma 4 12B (100-prompt lens)
logs/        console output of every run (download, smoke, fit, facts, refusal); local paths removed
figures/     en/ and de/, PNG + SVG, built by scripts/make_figures.py
```

## Glossary

| key | meaning |
|---|---|
| `variant` | `sft`, `heretic`, `basis` (= base model), `gemma4` |
| `kat` | item category: `fiktiv` fictional, `fakt` fact, `mathe` math |
| `q` / `antwort` | question / answer text (fact battery only) |
| `richtig` | automatic verdict is correct (bool) |
| `urteil` | automatic verdict: `nichtwissen` abstained, `konfabuliert` confabulated, `richtig` correct, `falsch` wrong |
| `praefix_ok` | re-tokenising prompt + answer reproduces the generated ids |
| `tokens[]` | one entry per answer token |
| `token` | the decoded answer token (fact battery only) |
| `layer.<L>` | lens read-out at layer L; `layer.out` = the model's own output |
| `H` | entropy in nats |
| `rang` | rank of the emitted token (0 = top-1) |
| `top3` | three most likely tokens with p (fact battery only) |
| `kipp` | lock-in layer: lowest fitted layer from which the emitted token stays rank 0 up to the top; `null` = only in the output |
| `sek` | seconds for this item |
| `daten.<set>.zeilen[]` | refusal pass, one row per prompt; `<set>` = `harmful` / `harmless` |
| `idx` | index into the dataset split |
| `abgelehnt` | refused, by keyword rule (`marker`) |
| `ids_ok` | the lens saw exactly the generated token ids |
| `n_tok` | number of answer tokens (≤ 32) |
| `H_out` | output entropy |

The console logs contain a few German status lines (`Geladen` loaded, `Gespeichert`
saved, `abgelehnt` refused, `FERTIG` done).
