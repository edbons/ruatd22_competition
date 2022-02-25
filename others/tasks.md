# Extract futures for Russian texts

Pipeline

1. Prepare dataset folders and files
2. No empath extraction
3. Q stat extraction (yule’s Q features (Lack of Coherence))
4. Other feauture extr
5. No coref (get_grid, get_coref)
6. No empath(get_empath)

Issues:
1. Divide by zero when no punctutation, no top pos (5 pos tags), no words in sent. Add 1e-8 to divider for stability
