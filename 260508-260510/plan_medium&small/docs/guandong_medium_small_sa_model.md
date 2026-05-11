# Medium And Small Planning

Current code structure:

- `run_yard_planner.py`: integrated entry point. If `--big-plan` is not provided, it calls `plan_big/block_allocation_main.py` first and summarizes the big plan into voyage-area quotas.
- `block_bay_planning/data_loader.py`: reads data and converts the big area plan into solver input.
- `block_bay_planning/block_plan_solver.py`: medium plan, assigning demand groups to yard blocks.
- `block_bay_planning/bay_plan_solver.py`: small plan, assigning large demand groups to bays inside selected blocks.
- `block_bay_planning/sa_solver.py`: orchestration, shared indexes, diagnostics, and CSV output.

Run the full chain:

```powershell
.\.venv\Scripts\python.exe "plan_medium&small\run_yard_planner.py" --data-dir 20260508data --output-dir outputs\block_bay_plan
```

Reuse an existing big plan CSV:

```powershell
.\.venv\Scripts\python.exe "plan_medium&small\run_yard_planner.py" --big-plan big_plan.csv --data-dir 20260508data --output-dir outputs\block_bay_plan
```

Accepted big-plan CSV formats:

- `voyage_id,area_no,planned_boxes`
- `voy_id,area_no,planned_qty`

Outputs:

Each run creates a new timestamped subfolder under `--output-dir`, so repeated runs do not overwrite previous results or fail because an old CSV is open in Excel.

- `big_plan.csv`: big-plan voyage-area-size quotas used by medium and small planning.
- `big_plan_detail.csv`: raw big-plan allocation detail when the full chain solves the big plan.
- `medium_plan.csv`: attribute groups assigned to yard blocks.
- `small_plan.csv`: large attribute groups assigned to bays.
- `diagnostics.json`: run summary and soft-constraint diagnostics.

Medium and small planning always allocate the exact volume supplied by the big plan. There is no separate window-box-ratio scaling parameter.
