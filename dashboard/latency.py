"""Latency dashboard: p50/p95 across sessions + waterfall of any single turn.

Run:  uv run streamlit run dashboard/latency.py
Reads every session log under logs/sessions (your live conversations) and logs/bench (benchmarks).
"""

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from server.config import settings  # noqa: E402
from server.metrics import STAGE_LABELS, STAGES, load_failures, load_turns  # noqa: E402

COLORS = ["#9aa3b2", "#7c5cff", "#2f6fed", "#3fb4e8", "#1a9b5c", "#d98a00"]
TARGET = settings.latency_target_ms

st.set_page_config(page_title="Voice latency", layout="wide")
st.title("Voice assistant — latency budget")
st.caption(f"Time from the user stopping talking to the first audio. Target: {TARGET} ms.")

DIRS = [ROOT / "logs" / "sessions", ROOT / "logs" / "faults", *sorted((ROOT / "logs" / "bench").glob("*"))]
rows, failures = [], []
for d in DIRS:
    if d.exists():
        rows += load_turns(d)
        failures += load_failures(d)
if not rows:
    st.info("No completed turns yet. Talk to the assistant or run scripts/benchmark.py.")
    st.stop()

df = pd.DataFrame(rows).drop(columns=["config"])
df["degraded"] = df["degraded"].fillna(False).astype(bool)
labels = sorted(df["label"].unique())
chosen = st.sidebar.multiselect("Run labels", labels, default=labels)
df = df[df["label"].isin(chosen)]

# ---------------------------------------------------------------- p50 / p95 table
st.subheader("p50 / p95 per stage")
keys = STAGES + ["total_ms", "perceived_ms"]
table = (
    df.groupby("label")[keys]
    .quantile([0.5, 0.95])
    .unstack()
    .round(0)
)
table.columns = [f"{STAGE_LABELS[k]} {'p50' if q == 0.5 else 'p95'}" for k, q in table.columns]
table.insert(0, "turns", df.groupby("label").size())
table.insert(1, "degraded", df.groupby("label")["degraded"].sum())
st.dataframe(table, use_container_width=True)
st.caption("Degraded turns (filler, fallback model, text-only...) count toward the totals but not the per-stage numbers.")

# ---------------------------------------------------------------- failures
st.subheader("Failures and what the assistant did instead")
fdf = pd.DataFrame(failures)
if fdf.empty:
    st.write("No failures recorded. Try the 🧪 Faults panel in the app, or `scripts/fault_injection.py`.")
else:
    fdf = fdf[fdf["label"].isin(chosen)]
    counts = fdf.groupby(["stage", "reason", "action"]).size().rename("count").reset_index()
    c1, c2 = st.columns([1, 2])
    c1.dataframe(counts, hide_index=True, use_container_width=True)
    c2.altair_chart(alt.Chart(counts).mark_bar().encode(
        x=alt.X("count:Q"), y=alt.Y("stage:N", title=None), color="reason:N",
        tooltip=["stage", "reason", "action", "count"]).properties(height=200), use_container_width=True)
    with st.expander("Every failure"):
        st.dataframe(fdf.sort_values("session_id", ascending=False), hide_index=True, use_container_width=True)

# ---------------------------------------------------------------- stacked bars per turn
st.subheader("Every turn, stacked by stage")
long = df.melt(id_vars=["label", "session_id", "turn_id"], value_vars=STAGES, var_name="stage", value_name="ms")
long["stage"] = long["stage"].map(STAGE_LABELS)
long["turn"] = long["label"] + " · " + long["session_id"].str[-6:] + " #" + long["turn_id"].astype(str)
stage_order = [STAGE_LABELS[k] for k in STAGES]
bars = alt.Chart(long).mark_bar().encode(
    x=alt.X("sum(ms):Q", title="ms"),
    y=alt.Y("turn:N", sort=None, title=None),
    color=alt.Color("stage:N", sort=stage_order, scale=alt.Scale(domain=stage_order, range=COLORS)),
    order=alt.Order("stage_idx:Q"),
    tooltip=["turn", "stage", alt.Tooltip("ms:Q", format=".0f")],
).transform_calculate(stage_idx=f"indexof({stage_order}, datum.stage)")
rule = alt.Chart(pd.DataFrame({"x": [TARGET]})).mark_rule(color="#d23c3c", strokeDash=[4, 4]).encode(x="x:Q")
st.altair_chart((bars + rule).properties(height=max(200, 18 * long["turn"].nunique())), use_container_width=True)

# ---------------------------------------------------------------- one turn waterfall
st.subheader("Waterfall of one turn")
df = df[~df["degraded"]]
df["key"] = df["label"] + " · " + df["session_id"] + " #" + df["turn_id"].astype(str)
pick = st.selectbox("Turn", df["key"].tolist()[::-1])
r = df[df["key"] == pick].iloc[0]
st.write(f"**Heard:** {r['transcript']}")
start, segs = 0.0, []
for k in STAGES:
    segs.append({"stage": STAGE_LABELS[k], "start": start, "end": start + max(0, r[k]), "ms": r[k]})
    start += max(0, r[k])
wf = alt.Chart(pd.DataFrame(segs)).mark_bar().encode(
    x=alt.X("start:Q", title="ms after the user stops talking"), x2="end:Q",
    y=alt.Y("stage:N", sort=stage_order, title=None),
    color=alt.Color("stage:N", sort=stage_order, scale=alt.Scale(domain=stage_order, range=COLORS), legend=None),
    tooltip=["stage", alt.Tooltip("ms:Q", format=".0f")],
)
st.altair_chart((wf + rule).properties(height=220), use_container_width=True)
st.metric("Total to first audio", f"{r['perceived_ms']:.0f} ms",
          delta=f"{r['perceived_ms'] - TARGET:+.0f} ms vs target", delta_color="inverse")
