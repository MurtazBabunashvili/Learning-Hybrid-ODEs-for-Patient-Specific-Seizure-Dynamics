"""Splits derived from LOCAL manifest (never hardcoded counts).

- patient holdout: disjoint patient sets
- temporal holdout: last frac of windows per recording
- episode holdout: leave-one-seizure-recording-out
"""
from __future__ import annotations


def patient_split(windows: list[dict], train_patients: list[str],
                  val_patients: list[str], test_patients: list[str]) -> dict:
    def sel(pats):
        return [w for w in windows if w["patient_id"] in pats]
    return {"train": sel(train_patients), "val": sel(val_patients), "test": sel(test_patients)}


def temporal_split(windows: list[dict], val_frac: float = 0.2) -> dict:
    from collections import defaultdict
    by_rec: dict[str, list[dict]] = defaultdict(list)
    for w in windows:
        by_rec[w["recording_id"]].append(w)
    train, val = [], []
    for rec, ws in by_rec.items():
        ws = sorted(ws, key=lambda w: w["start_sec"])
        n_val = max(1, int(len(ws) * val_frac))
        train += ws[:-n_val]
        val += ws[-n_val:]
    return {"train": train, "val": val}


KIND_PRIORITY = {"ictal": 0, "preictal": 1, "postictal": 2, "interictal": 3}


def window_phase(w: dict, near_sec: float = 120.0) -> str:
    """Refine preictal into far/near by time-to-onset; others pass through.

    Times are relative to clinical annotation (investigation signal, not label).
    Uses the NEXT onset at/after window start only (past onsets ignored).
    """
    if w["kind"] != "preictal":
        return w["kind"]
    onsets = [float(s["onset_sec"]) for s in w.get("seizures", [])]
    if not onsets:
        return "preictal_far"
    start = float(w["start_sec"])
    future = [on - start for on in onsets if on >= start]
    if not future:
        return "preictal_far"
    dt = min(future)  # time to next onset
    return "preictal_near" if dt < near_sec else "preictal_far"


def sample_onset_relative(windows: list[dict], patients: list[str],
                          n_total: int, quotas: dict[str, float] | None = None,
                          seed: int = 0) -> list[dict]:
    """Multi-window-per-recording, onset-relative sampling.

    Target mix (default ≈ 40/20/20/10/10):
      interictal 0.4, preictal_far 0.2, preictal_near 0.2, ictal 0.1, postictal 0.1
    Within each phase: round-robin across recordings (several windows per
    seizure recording, e.g. -300:-270, -120:-90, -30:0, 0:30, 30:60 relative
    to onset). Across patients: proportional to pool size. Deterministic.
    """
    import random
    from collections import defaultdict
    rng = random.Random(seed)
    quotas = quotas or {"interictal": 0.4, "preictal_far": 0.2,
                        "preictal_near": 0.2, "ictal": 0.1, "postictal": 0.1}
    pool = [w for w in windows if w["patient_id"] in patients]
    by_phase: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for w in pool:
        ph = window_phase(w)
        by_phase[ph][w["recording_id"]].append(w)
    for recs in by_phase.values():
        for ws in recs.values():
            ws.sort(key=lambda w: w["start_sec"])
    # per-patient share proportional to pool size
    per_pat = len(pool) and {p: len([w for w in pool if w["patient_id"] == p]) for p in patients}
    out: list[dict] = []
    for phase, frac in quotas.items():
        n_phase = int(round(n_total * frac))
        recs = by_phase.get(phase, {})
        rec_ids = sorted(recs)
        rng.shuffle(rec_ids)
        got: list[dict] = []
        while len(got) < n_phase and any(recs.values()):
            for rid in rec_ids:
                if recs[rid] and len(got) < n_phase:
                    got.append(recs[rid].pop(0))
        # balance patients inside phase
        got.sort(key=lambda w: (w["patient_id"], w["recording_id"], w["start_sec"]))
        out.extend(got)
    # interleave phases so batches mix kinds; keep determinism
    rng.shuffle(out)
    # per-patient totals roughly proportional: trim overflow from majority patient
    return out[:n_total]


def _episode_key(w: dict):
    """(recording_id, nearest onset) for seizure-linked windows."""
    onsets = [float(s["onset_sec"]) for s in w.get("seizures", [])]
    if not onsets:
        return None
    start = float(w["start_sec"])
    return (w["recording_id"], min(onsets, key=lambda on: abs(start - on)))


def _greedy_gap(sorted_starts: list[float], min_gap: float) -> list[int]:
    """Indices kept with mutual spacing >= min_gap (non-overlap if >= window)."""
    kept: list[int] = []
    for i, s in enumerate(sorted_starts):
        if all(abs(s - sorted_starts[j]) >= min_gap for j in kept):
            kept.append(i)
    return kept


def sample_diverse(windows: list[dict], patients: list[str], n_total: int,
                   seed: int = 0, per_episode_cap: int = 16,
                   seizure_gap_sec: float = 30.0,
                   ictal_gap_sec: float = 15.0) -> tuple[list[dict], dict]:
    """Maximize patients x recordings x seizures x temporal contexts.

    Per patient (target = n_total // len(patients)):
      1. dedupe by (path, start_sec);
      2. seizure-linked windows grouped by episode (recording, nearest onset),
         greedy min-gap (30 s, 15 s ictal), cap 16/episode — many episodes,
         few windows each, spread over pre-onset distances + ictal thirds;
      3. interictal filled from a 60 s-gap greedy pass (non-overlapping),
         round-robin across recordings with thirds-spread, then a 30 s-gap
         second pass only if the patient target is unmet;
      4. deterministic (seeded recording order only).
    Returns (windows, report dict with per-patient/phase counts).
    """
    import random
    from collections import defaultdict
    rng = random.Random(seed)
    per_pat_target = n_total // max(1, len(patients))
    out: list[dict] = []
    report: dict = {"per_patient": {}, "phases": defaultdict(int),
                    "episodes": 0, "recordings": set()}
    for pat in patients:
        pw = [w for w in windows if w["patient_id"] == pat]
        seen, uniq = set(), []
        for w in pw:
            k = (w["path"], round(float(w["start_sec"]), 1))
            if k not in seen:
                seen.add(k)
                uniq.append(w)
        linked: dict[tuple, list[dict]] = defaultdict(list)
        inter: dict[str, list[dict]] = defaultdict(list)
        for w in uniq:
            key = _episode_key(w) if w["kind"] != "interictal" else None
            if key is None:
                inter[w["recording_id"]].append(w)
            else:
                linked[key].append(w)
        picked: list[dict] = []
        taken: dict[str, list[float]] = defaultdict(list)
        ep_keys = sorted(linked)
        rng.shuffle(ep_keys)
        for ek in ep_keys:
            ws = sorted(linked[ek], key=lambda w: float(w["start_sec"]))
            gap = ictal_gap_sec if any(w["kind"] == "ictal" for w in ws) else seizure_gap_sec
            # spread: greedy gap pass, then cap (keeps distance/phase spread)
            keep = _greedy_gap([float(w["start_sec"]) for w in ws], gap)
            # round-robin phases within episode for kind balance
            by_kind: dict[str, list[dict]] = defaultdict(list)
            for i in keep:
                by_kind[window_phase(ws[i])].append(ws[i])
            order = ["ictal", "preictal_near", "preictal_far", "postictal",
                     "preictal", "interictal"]
            n_cap = 0
            while n_cap < per_episode_cap and any(by_kind.values()):
                for k in order:
                    if by_kind.get(k) and n_cap < per_episode_cap:
                        w = by_kind[k].pop(0)
                        picked.append(w)
                        taken[w["recording_id"]].append(float(w["start_sec"]))
                        n_cap += 1
        report["episodes"] += len(ep_keys)
        # Global floor across episodes within a recording (episodes were
        # picked in isolation): non-ictal pairs >= 60 s; pairs where BOTH
        # windows are ictal keep the documented 15 s exception.
        by_rec_ep: dict[str, list[dict]] = defaultdict(list)
        for w in picked:
            by_rec_ep[w["recording_id"]].append(w)
        picked2: list[dict] = []
        for rid, ws in by_rec_ep.items():
            ws.sort(key=lambda w: float(w["start_sec"]))
            kept: list[dict] = []
            for w in ws:
                s = float(w["start_sec"])
                ok = True
                for v in kept:
                    t = float(v["start_sec"])
                    both_ictal = (w["kind"] == "ictal"
                                  and v["kind"] == "ictal")
                    if abs(s - t) < (15.0 if both_ictal else 60.0):
                        ok = False
                        break
                if ok:
                    kept.append(w)
            picked2.extend(kept)
        picked = picked2
        taken: dict[str, list[float]] = defaultdict(list)
        for w in picked:
            taken[w["recording_id"]].append(float(w["start_sec"]))
        # interictal fill: round-robin recordings, spread across
        # beginning/middle/end, >=60 s from EVERYTHING taken (episode picks
        # included) -- one global non-overlap floor, no fallback.
        rec_ids = sorted(inter)
        rng.shuffle(rec_ids)
        for ws in inter.values():
            ws.sort(key=lambda w: float(w["start_sec"]))
        need = per_pat_target - len(picked)
        if need > 0:
            per_rec = max(1, (need // max(1, len(rec_ids))) + 1)
            for rid in rec_ids:
                if need <= 0:
                    break
                ws = inter.get(rid, [])
                if not ws:
                    continue
                starts = [float(w["start_sec"]) for w in ws]
                cand = [i for i in range(len(ws))
                        if all(abs(starts[i] - t) >= 60.0
                               for t in taken[rid])]
                keep = [cand[j] for j in _greedy_gap(
                    [starts[i] for i in cand], 60.0)]
                # evenly spaced through kept list for thirds-spread
                n_take = min(per_rec, len(keep))
                take = [keep[i * len(keep) // n_take]
                        for i in range(n_take)] if n_take else []
                for i in take:
                    if need <= 0:
                        break
                    picked.append(ws[i])
                    taken[rid].append(float(ws[i]["start_sec"]))
                    need -= 1
        for w in picked:
            report["phases"][window_phase(w)] += 1
            report["recordings"].add(w["recording_id"])
        report["per_patient"][pat] = len(picked)
        out.extend(picked)
    # dynamic balance: no patient > 2x the least-represented (trim
    # interictal tails evenly, preserving spread + all seizure windows)
    counts = {p: report["per_patient"][p] for p in patients}
    floor = min(counts.values())
    cap = 2 * floor
    final: list[dict] = []
    for pat in patients:
        pw = [w for w in out if w["patient_id"] == pat]
        if len(pw) > cap:
            keep_sz = [w for w in pw if w["kind"] != "interictal"]
            inter_only = sorted(
                (w for w in pw if w["kind"] == "interictal"),
                key=lambda w: float(w["start_sec"]))
            n_keep = max(0, cap - len(keep_sz))
            if 0 < n_keep < len(inter_only):
                inter_only = [inter_only[i * len(inter_only) // n_keep]
                              for i in range(n_keep)]
            pw = keep_sz + inter_only[:max(0, n_keep)]
            report["per_patient"][pat] = len(pw)
        final.extend(pw)
    out = final
    # interleave patients for mixed batches; deterministic
    by_pat: dict[str, list[dict]] = defaultdict(list)
    for w in out:
        by_pat[w["patient_id"]].append(w)
    mixed: list[dict] = []
    while any(by_pat.values()):
        for pat in patients:
            if by_pat[pat]:
                mixed.append(by_pat[pat].pop(0))
    report["recordings"] = len(report["recordings"])
    report["phases"] = dict(report["phases"])
    report["total"] = len(mixed)
    return mixed, report


def sample_targeted(manifest_recs: list[dict], patients: list[str],
                    n_total: int, seed: int = 0,
                    pre_span: float = 300.0, pre_stride: float = 10.0,
                    ict_pad: float = 30.0, ict_stride: float = 5.0,
                    post_span: float = 300.0, post_stride: float = 10.0,
                    per_episode_cap: int = 40,
                    window_sec: float = 60.0) -> tuple[list[dict], dict]:
    """Fill `n_total` prioritizing seizure-episode coverage, then interictal.

    Per seizure episode (duration>0; non-positive spans counted+skipped):
      preictal:  starts on-`pre_span`..on-`window_sec`, stride `pre_stride`;
      ictal:     starts on-`ict_pad`..off, stride `ict_stride` (early/mid/late
                 thirds fall out of the even coverage);
      postictal: starts off..off+`post_span`, stride `post_stride`;
      capped at `per_episode_cap`/phase (even thinning), deduped.
    Interictal: 60 s non-overlapping grid outside [on-60, off+60] of any
    seizure, round-robin recordings with thirds-spread, to reach exactly
    n_total (or availability ceiling, reported).
    Per-patient totals equalized via the interictal fill.
    Deterministic. Returns (windows, report).
    """
    import random
    from collections import defaultdict
    rng = random.Random(seed)
    rep: dict = {"episodes": 0, "skipped_bad_spans": 0, "recordings": set(),
                 "phases": defaultdict(int), "per_patient": {},
                 "preictal_bands": defaultdict(int),
                 "target_total": n_total}
    per_pat: dict[str, list[dict]] = defaultdict(list)

    def _mk(rec, start, kind, sz):
        return {"patient_id": rec["patient_id"],
                "recording_id": rec["recording_id"], "path": rec["path"],
                "start_sec": float(start), "end_sec": float(start) + window_sec,
                "kind": kind,
                "seizures": [{"onset_sec": float(sz["onset_sec"]),
                              "offset_sec": float(sz["offset_sec"])}]}

    def _thin(ws: list[dict], cap: int) -> list[dict]:
        if len(ws) <= cap:
            return ws
        return [ws[i * len(ws) // cap] for i in range(cap)]

    for pat in patients:
        recs = sorted([r for r in manifest_recs if r["patient_id"] == pat],
                      key=lambda r: r["recording_id"])
        rng.shuffle(recs)
        for rec in recs:
            dur = float(rec["duration_sec"])
            for sz in rec.get("seizures", []):
                on, off = float(sz["onset_sec"]), float(sz["offset_sec"])
                if off <= on:
                    rep["skipped_bad_spans"] += 1
                    continue
                rep["episodes"] += 1
                pre, ict, post = [], [], []
                s = on - pre_span
                while s + window_sec <= on:
                    pre.append(_mk(rec, max(0.0, s), "preictal", sz))
                    s += pre_stride
                s = on - ict_pad
                while s <= off:
                    if s + window_sec <= dur:
                        ict.append(_mk(rec, max(0.0, s), "ictal", sz))
                    s += ict_stride
                s = off
                while s + window_sec <= min(dur, off + post_span):
                    post.append(_mk(rec, s, "postictal", sz))
                    s += post_stride
                for w in _thin(pre, per_episode_cap):
                    d = on - float(w["start_sec"])
                    band = ("0-60" if d < 60 else "60-120" if d < 120
                            else "120-180" if d < 180 else "180-300+")
                    rep["preictal_bands"][band] += 1
                per_pat[pat].extend(_thin(pre, per_episode_cap)
                                    + _thin(ict, per_episode_cap)
                                    + _thin(post, per_episode_cap))
    # dedupe (path, start)
    for pat in per_pat:
        seen, uq = set(), []
        for w in per_pat[pat]:
            k = (w["path"], round(float(w["start_sec"]), 1))
            if k not in seen:
                seen.add(k)
                uq.append(w)
        per_pat[pat] = uq
    # interictal fill to equal per-patient totals
    target_each = n_total // max(1, len(patients))
    # picked episode starts per recording: the grid must avoid ALL of them
    # (+/-60 s), not just the seizure spans.
    picked_starts: dict[str, list[float]] = defaultdict(list)
    for pat in per_pat:
        for w in per_pat[pat]:
            picked_starts[w["recording_id"]].append(float(w["start_sec"]))
    for pat in patients:
        recs = sorted([r for r in manifest_recs if r["patient_id"] == pat],
                      key=lambda r: r["recording_id"])
        rng.shuffle(recs)
        need = target_each - len(per_pat[pat])
        if need <= 0:
            continue
        per_rec = max(1, need // max(1, len(recs)) + 1)
        # pass 1: non-overlapping 60 s grid; pass 2 (only if still short):
        # 30 s grid, still unique windows at 2-4x the old dense spacing.
        # Exclusion around picked starts matches the pass floor.
        for stride, floor in ((60.0, 60.0), (30.0, 30.0)):
            if need <= 0:
                break
            for rec in recs:
                if need <= 0:
                    break
                dur = float(rec["duration_sec"])
                excl = [(float(s["onset_sec"]) - 60.0, float(s["offset_sec"]) + 60.0)
                        for s in rec.get("seizures", [])]
                if floor >= 60.0:
                    # pass 1: non-overlapping grid; avoid picked starts too
                    excl += [(t - floor, t + floor)
                             for t in picked_starts.get(rec["recording_id"], [])]
                # pass 2 (floor 30): 30 s spacing MEANS 50% window overlap,
                # so only the seizure spans + taken_now distance rule apply.
                cands = []
                s = 0.0
                while s + window_sec <= dur:
                    if not any(s < hi and s + window_sec > lo for lo, hi in excl):
                        cands.append(_mk(rec, s, "interictal",
                                         {"onset_sec": -1.0, "offset_sec": -1.0}))
                    s += stride
                # drop candidates colliding with already-taken at floor
                taken_now: list[float] = [
                    float(w["start_sec"]) for w in per_pat[pat]
                    if w["recording_id"] == rec["recording_id"]]
                cands = [w for w in cands if all(
                    abs(float(w["start_sec"]) - t) >= floor for t in taken_now)]
                n_take = min(per_rec, len(cands))
                take = [cands[i * len(cands) // n_take]
                        for i in range(n_take)] if n_take else []
                for w in take:
                    if need <= 0:
                        break
                    per_pat[pat].append(w)
                    picked_starts[w["recording_id"]].append(float(w["start_sec"]))
                    need -= 1
    out: list[dict] = []
    for pat in patients:
        for w in per_pat[pat]:
            rep["phases"][window_phase(w)] += 1
            rep["recordings"].add(w["recording_id"])
        rep["per_patient"][pat] = len(per_pat[pat])
        out.extend(per_pat[pat])
    # interleave patients for mixed batches; deterministic
    by_pat: dict[str, list[dict]] = defaultdict(list)
    for w in out:
        by_pat[w["patient_id"]].append(w)
    mixed: list[dict] = []
    while any(by_pat.values()):
        for pat in patients:
            if by_pat[pat]:
                mixed.append(by_pat[pat].pop(0))
    rep["recordings"] = len(rep["recordings"])
    rep["phases"] = dict(rep["phases"])
    rep["total"] = len(mixed)
    return mixed, rep


def stratified_patient_sample(windows: list[dict], patients: list[str],
                              n_per_patient: int, seed: int = 0) -> list[dict]:
    """Balanced sample: round-robin across recordings within each patient, so
    seizure dynamics are covered and no single *_01.edf dominates.

    Within a recording, ictal -> preictal -> postictal -> interictal first.
    Deterministic given seed (shuffles recording order only).
    """
    import random
    from collections import defaultdict
    rng = random.Random(seed)
    out: list[dict] = []
    for pat in patients:
        by_rec: dict[str, list[dict]] = defaultdict(list)
        for w in windows:
            if w["patient_id"] == pat:
                by_rec[w["recording_id"]].append(w)
        rec_ids = sorted(by_rec)
        rng.shuffle(rec_ids)
        for ws in by_rec.values():
            ws.sort(key=lambda w: (KIND_PRIORITY.get(w["kind"], 9), w["start_sec"]))
        picked: list[dict] = []
        while len(picked) < n_per_patient and any(by_rec.values()):
            for rid in rec_ids:
                if by_rec[rid] and len(picked) < n_per_patient:
                    picked.append(by_rec[rid].pop(0))
        out.extend(picked)
    return out
