# tb_bridge

Bypass console logs -> TensorBoard under `/root/tf-logs` (AutoDL AutoPanel port 6007).

- Parses every numeric `k=v` on `Step N:` lines as `train/<k>` (plus top-level `loss`).
- Between `log_interval` Steps, carries the latest full metric set onto each Progress `it`.
- Also writes `progress/it` and `timing/*`. Does not touch the training process.
