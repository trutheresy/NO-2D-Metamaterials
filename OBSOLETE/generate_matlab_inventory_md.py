from __future__ import annotations

import ast
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
OUT_MD = REPO_ROOT / "MATLAB_IO_AND_M_FILES_INVENTORY.md"


PY_PATTERNS = {
    "reads_mat": re.compile(r"loadmat\(|sio\.loadmat|mat73|h5py\.File\(", re.IGNORECASE),
    "writes_mat": re.compile(r"savemat\(|sio\.savemat|h5py\.File\(", re.IGNORECASE),
    "calls_matlab": re.compile(r"matlab\s*-batch|matlab\.engine|subprocess.*matlab", re.IGNORECASE),
    "matlab_struct_names": re.compile(
        r"\b(WAVEVECTOR_DATA|EIGENVALUE_DATA|EIGENVECTOR_DATA|CONSTITUTIVE_DATA)\b"
    ),
}


def _rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1", errors="ignore")


def _py_summary(path: Path, text: str) -> str:
    try:
        mod = ast.parse(text)
        doc = ast.get_docstring(mod)
        if doc:
            line = doc.strip().splitlines()[0].strip()
            if line:
                return line
    except Exception:
        pass

    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#") and len(s.strip("# ").strip()) > 0:
            return s.strip("# ").strip()
    return "No top-level docstring/comment; likely utility or bridge script."


def _m_summary(path: Path, text: str) -> str:
    lines = text.splitlines()
    for line in lines[:40]:
        s = line.strip()
        if s.startswith("%"):
            val = s.lstrip("%").strip()
            if val:
                return val
    if lines:
        first = lines[0].strip()
        if first.lower().startswith("function"):
            return "MATLAB function file (no header comment)."
    return "MATLAB script/function (no header comment)."


def collect_python_matlab_io() -> list[dict]:
    rows: list[dict] = []
    for path in REPO_ROOT.rglob("*.py"):
        if ".git/" in _rel(path):
            continue
        text = _read_text(path)
        flags = {k: bool(rx.search(text)) for k, rx in PY_PATTERNS.items()}
        if not any(flags.values()):
            continue

        roles = []
        if flags["reads_mat"]:
            roles.append("reads .mat/HDF5-MAT")
        if flags["writes_mat"]:
            roles.append("writes .mat/HDF5-MAT")
        if flags["calls_matlab"]:
            roles.append("invokes MATLAB runtime")
        if flags["matlab_struct_names"]:
            roles.append("uses MATLAB-style dataset field names")

        rows.append(
            {
                "path": _rel(path),
                "roles": ", ".join(roles) if roles else "MATLAB-adjacent logic",
                "summary": _py_summary(path, text),
                "scope": "obsolete" if _rel(path).startswith("obsolete/") else "active",
            }
        )
    rows.sort(key=lambda r: (r["scope"], r["path"]))
    return rows


def collect_matlab_files() -> list[dict]:
    rows: list[dict] = []
    for path in REPO_ROOT.rglob("*.m"):
        if ".git/" in _rel(path):
            continue
        text = _read_text(path)
        rows.append(
            {
                "path": _rel(path),
                "summary": _m_summary(path, text),
                "scope": "obsolete" if _rel(path).startswith("obsolete/") else "active",
            }
        )
    rows.sort(key=lambda r: (r["scope"], r["path"]))
    return rows


def render_md(py_rows: list[dict], m_rows: list[dict]) -> str:
    active_py = [r for r in py_rows if r["scope"] == "active"]
    obsolete_py = [r for r in py_rows if r["scope"] == "obsolete"]
    active_m = [r for r in m_rows if r["scope"] == "active"]
    obsolete_m = [r for r in m_rows if r["scope"] == "obsolete"]

    out: list[str] = []
    out.append("# MATLAB I/O and MATLAB-Format Script Inventory")
    out.append("")
    out.append(
        "This inventory lists (1) non-MATLAB scripts that ingest/emit MATLAB formats or invoke MATLAB, "
        "and (2) every MATLAB `.m` script/function in the repository."
    )
    out.append("")
    out.append("## Counts")
    out.append("")
    out.append(f"- Python MATLAB-I/O/bridge scripts (active): `{len(active_py)}`")
    out.append(f"- Python MATLAB-I/O/bridge scripts (obsolete): `{len(obsolete_py)}`")
    out.append(f"- MATLAB `.m` files (active): `{len(active_m)}`")
    out.append(f"- MATLAB `.m` files (obsolete): `{len(obsolete_m)}`")
    out.append("")
    out.append("## Python Scripts With MATLAB Inputs/Outputs or MATLAB Bridge Behavior")
    out.append("")
    out.append("| Scope | Script | MATLAB Interaction | What It Does |")
    out.append("|---|---|---|---|")
    for r in py_rows:
        out.append(f"| {r['scope']} | `{r['path']}` | {r['roles']} | {r['summary'].replace('|', '/')} |")
    out.append("")
    out.append("## All MATLAB `.m` Files")
    out.append("")
    out.append("| Scope | MATLAB File | What It Does |")
    out.append("|---|---|---|")
    for r in m_rows:
        out.append(f"| {r['scope']} | `{r['path']}` | {r['summary'].replace('|', '/')} |")
    out.append("")
    out.append("## Notes")
    out.append("")
    out.append("- `active` means outside `obsolete/`; `obsolete` means already archived under `obsolete/`.")
    out.append("- This is a static code scan based on file extension and MATLAB-related patterns.")
    return "\n".join(out) + "\n"


def main() -> None:
    py_rows = collect_python_matlab_io()
    m_rows = collect_matlab_files()
    md = render_md(py_rows, m_rows)
    OUT_MD.write_text(md, encoding="utf-8")
    print(str(OUT_MD))
    print(f"python_rows={len(py_rows)} m_rows={len(m_rows)}")


if __name__ == "__main__":
    main()

