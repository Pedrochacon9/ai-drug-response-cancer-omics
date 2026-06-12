## Dataset Summary (IMPROVE / DeepTTC) — Normal Split (New Pairs)

> Summary calculated from `exp_result/` using the standard cell–drug pair split, with no overlapping pairs between train/val/test.

| Split | #samples | #cells | #drugs |
| ----- | -------: | -----: | -----: |
| train |     7616 |    411 |     24 |
| val   |      952 |    371 |     24 |
| test  |      951 |    371 |     24 |

**Entity Overlap**

* Cells: train∩val = 371, train∩test = 371, val∩test = 333
* Drugs: train∩val = 24, train∩test = 24, val∩test = 24

**Sample Overlap**

* Pairs `(cell, drug)`: **0** overlap between splits

**Interpretation**

* Although **cells** and **drugs** are repeated across splits, no `(cell, drug)` pair is repeated. Therefore, this setting evaluates generalization to **new combinations**, not to completely unseen entities.
* This is why more challenging splits are useful, such as `drugout_seed42` for unseen drugs and `cellout_seed42` for unseen cells.
