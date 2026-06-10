## Dataset summary (IMPROVE / DeepTTC) — split “normal” (pares nuevos)

> Resumen calculado sobre `exp_result/` (split estándar por pares cell–drug, sin solape de pares entre train/val/test).

| Split | #ejemplos | #cells | #drugs |
|---|---:|---:|---:|
| train | 7616 | 411 | 24 |
| val   | 952  | 371 | 24 |
| test  | 951  | 371 | 24 |

**Solapamiento (entidades)**
- Cells: train∩val=371, train∩test=371, val∩test=333  
- Drugs: train∩val=24, train∩test=24, val∩test=24  

**Solapamiento (ejemplos)**
- Pairs (cell,drug): **0** solapamiento entre splits

**Interpretación**
- Aunque se repitan **células** y **fármacos** entre splits, **no se repite ningún par (cell,drug)**: se evalúa generalización a **combinaciones nuevas**, no a entidades completamente nuevas.
- Por eso son útiles los splits más exigentes: `drugout_seed42` (fármacos no vistos) y `cellout_seed42` (células no vistas).