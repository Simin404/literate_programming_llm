
#!/usr/bin/env python3
import sys
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import pandas as pd
except ImportError:
    pd = None

MODEL_LINE_RE = re.compile(
    r"""^(?P<ts>[\d\-:\s,]+)\s*-\s*Model:\s*(?P<model>[^;]+);\s*Classifier:\s*(?P<classifier>[^;]+);\s*Parameters:\s*(?P<params>.+?)\s*$"""
)

LANG_RE = re.compile(r"""Language:\s*Accuracy\s*->\s*(?P<val>[0-9.]+)""")
TASK_RE = re.compile(r"""Task:\s*Accuracy\s*->\s*(?P<val>[0-9.]+)""")
FINAL_RE = re.compile(
    r"""Final\s+Accuracy\s*->\s*Language:\s*(?P<lang>[0-9.]+),\s*Task:\s*(?P<task>[0-9.]+),\s*Desc:\s*(?P<desc>[0-9.]+)"""
)

def _parse_params(text: str) -> str:
    """
    Try to normalize the Parameters value to JSON if possible.
    Returns a JSON string or the original text if parsing fails.
    """
    t = text.strip()
    # Common cases: "default" or "{'n_neighbors': 10}"
    if t.lower() == "default":
        return json.dumps({"default": True})
    # Convert single quotes to double quotes for JSON
    if t.startswith("{") and t.endswith("}"):
        jlike = t.replace("'", '"')
        try:
            parsed = json.loads(jlike)
            return json.dumps(parsed, separators=(",", ":"))
        except Exception:
            pass
    return t  # fallback (raw string)

def parse_log_lines(lines: List[str]) -> List[Dict[str, Any]]:
    """
    Single pass stream parser.
    Creates a record when a 'Final Accuracy -> ...' line is observed,
    using the most recent model/params + current language/task context.
    """
    records: List[Dict[str, Any]] = []

    cur_model: Optional[str] = None
    cur_classifier: Optional[str] = None
    cur_params_raw: Optional[str] = None
    cur_ts_start: Optional[str] = None

    # temp values seen since the current model line
    cur_lang_seen: Optional[float] = None
    cur_task_seen: Optional[float] = None

    for raw in lines:
        line = raw.rstrip("\n")

        # New run starts when we hit a "Model:" line
        m = MODEL_LINE_RE.match(line)
        if m:
            cur_ts_start = m.group("ts").strip()
            cur_model = m.group("model").strip()
            cur_classifier = m.group("classifier").strip()
            cur_params_raw = _parse_params(m.group("params"))
            cur_lang_seen = None
            cur_task_seen = None
            continue

        # Track intermediate language/task accuracy if present
        ml = LANG_RE.search(line)
        if ml:
            try:
                cur_lang_seen = float(ml.group("val"))
            except Exception:
                pass

        mt = TASK_RE.search(line)
        if mt:
            try:
                cur_task_seen = float(mt.group("val"))
            except Exception:
                pass

        # Emit a record when Final Accuracy line appears
        mf = FINAL_RE.search(line)
        if mf and cur_model is not None:
            lang_final = float(mf.group("lang"))
            task_final = float(mf.group("task"))
            desc_final = float(mf.group("desc"))

            rec = {
                "timestamp_start": cur_ts_start,
                "model": cur_model,
                "classifier": cur_classifier,
                "parameters": cur_params_raw,
                "language_acc": lang_final if mf else cur_lang_seen,
                "task_acc": task_final if mf else cur_task_seen,
                "desc": desc_final,
            }
            records.append(rec)

            # after final, we keep context until next Model line
            # (some logs may show multiple finals per model; that's okay)

    return records

def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_knn_logs.py <path-to-log-file> [--csv out.csv]")
        print("Hint: python parse_knn_logs.py knn_runs.log --csv results.csv")
        sys.exit(1)

    log_path = Path(sys.argv[1])
    if not log_path.exists():
        print(f"Error: file not found: {log_path}")
        sys.exit(2)

    out_csv = None
    if "--csv" in sys.argv:
        try:
            out_csv = sys.argv[sys.argv.index("--csv") + 1]
        except Exception:
            print("Error: --csv flag provided but no filename given")
            sys.exit(3)

    with log_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    records = parse_log_lines(lines)

    if not records:
        print("No records parsed. Please check the log format or regexes in the script.")
        sys.exit(0)

    # Print a quick console summary
    print(f"Parsed {len(records)} runs.")
    # Show a couple of sample rows
    for r in records[:5]:
        print(r)

    # Save CSV if requested
    if out_csv:
        if pd is None:
            # Write a simple CSV manually (without pandas)
            import csv
            with open(out_csv, "w", newline="", encoding="utf-8") as fp:
                writer = csv.DictWriter(fp, fieldnames=list(records[0].keys()))
                writer.writeheader()
                writer.writerows(records)
        else:
            df = pd.DataFrame.from_records(records)
            df.to_csv(out_csv, index=False)
        print(f"Saved CSV to: {out_csv}")

    # If pandas is available and not writing to CSV, we can still show a preview
    if pd is not None and not out_csv:
        df = pd.DataFrame.from_records(records)
        # Try a sensible column order
        cols = ["timestamp_start", "model", "classifier", "parameters", "language_acc", "task_acc", "desc"]
        df = df[cols]
        try:
            from ace_tools import display_dataframe_to_user
            display_dataframe_to_user("Parsed KNN Log Results", df)
        except Exception:
            # fallback to printing
            print(df.head(20).to_string(index=False))

if __name__ == "__main__":
    main()
