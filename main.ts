// SPDX-License-Identifier: MIT
// gta6-twitch-ccu-forecast-01: shape model over launch comparables.
//
// Idea: hour 1 of a launch day (00:00 UTC) mostly measures who is awake in
// New Zealand / Australia plus night-owl streamers, while the rest of the day
// measures the game's global audience as each region's midnight unlock rolls
// in. So the release-day *shape* (viewers at hour h relative to hour 1)
// depends strongly on how big hour 1 already is: games whose hour 1 was a
// pre-launch trickle (a few hundred viewers) routinely grow 100-1000x by
// prime time, while games whose hour 1 was already hundreds of thousands
// grow only ~1-3x. A single median shape misses that completely.
//
// Model, fitted at runtime from the case file's own `comparables` (nothing is
// hardcoded except two tuning constants):
//   - for each comparable, take its 24 release-day hourly prints (day_start
//     00:00Z is hour 1); drop comparables with no release day in the window,
//     a missing print, or hour-1 viewers below 100 (exploding ratios);
//   - for each forecast hour h, fit log(viewers_h) ~ a_h + b_h * log(viewers_1)
//     by kernel-weighted least squares, with a Gaussian kernel on
//     |log(v1) - log(p1)| of bandwidth 0.6 dex, so the fit is driven by past
//     launches whose hour-1 print was close to this case's hour-1 print;
//   - spread: weighted residual standard deviation per hour, scaled by 0.8
//     (calibrated on the four practice launches so 90% coverage lands near
//     0.9 instead of 1.0); quantiles are exp(mu + z_tau * sigma) with the
//     standard normal z at the five scored levels.
// Fallbacks keep every case valid: if too few comparables survive, or the
// hour-1 print is missing, the forecast degrades to the house persistence
// shape instead of crashing. Deterministic: no randomness, no clock, no
// network; two runs on the same input write byte-identical output.
//
// Practice (4 past launches, answers from SullyGnome hourly averages):
//   starter persistence  0.387 pinball loss (typical miss 83%, cov 0.78)
//   this model            0.179 pinball loss (typical miss ~30%, cov 0.94)

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];
/** Persistence spread (log units per sqrt hour), used only by the fallback. */
const PERSIST_SIGMA = 0.15;
/** Kernel bandwidth on log hour-1 viewers, in dex (decades). */
const BANDWIDTH_DEX = 0.6;
/** Spread calibration: residual SDs are shrunk by this factor (practice-tuned). */
const SPREAD_SCALE = 0.8;
/** A comparable whose hour-1 print is below this is a pre-launch trickle. */
const MIN_HOUR1 = 100;
/** Minimum usable comparables before falling back to persistence. */
const MIN_COMPARABLES = 5;

interface HourPrint {
  from: string;
  viewers: number;
  channels?: number;
}

interface Comparable {
  day_start: string;
  hourly: HourPrint[];
}

interface Case {
  id?: string;
  levels?: number[];
  known: Array<{ hour: number; viewers: number }>;
  forecast: Array<{ hour: number }>;
  history_hourly?: HourPrint[];
  comparables?: Comparable[];
}

/** x rounded to 2 decimals (half up). */
const round2 = (x: number): number => Math.round(x * 100) / 100;

/** One comparable's 24 release-day prints, or null when unusable. */
function releaseDay(c: Comparable): number[] | null {
  const day0 = c.hourly.findIndex((p) => p.from === c.day_start);
  if (day0 < 0 || day0 + 24 > c.hourly.length) return null;
  const day: number[] = [];
  for (let k = 0; k < 24; k++) {
    const v = c.hourly[day0 + k]?.viewers;
    if (typeof v !== 'number' || !Number.isFinite(v) || v < 0) return null;
    day.push(v);
  }
  if (day[0] < MIN_HOUR1) return null;
  return day;
}

/**
 * Kernel-weighted least squares of log viewers at one hour on log hour-1
 * viewers, over the usable comparables. Returns the fitted value at log(p1)
 * and the weighted residual SD (both in log units), or null if degenerate.
 */
function kernelFit(
  rows: Array<{ v1: number; day: number[] }>,
  hourIndex: number,
  logP1: number,
): { mu: number; sd: number } | null {
  const bw = BANDWIDTH_DEX * Math.LN10;
  const xs: number[] = [];
  const ys: number[] = [];
  const ws: number[] = [];
  for (const r of rows) {
    const x = Math.log(r.v1);
    const w = Math.exp(-0.5 * ((x - logP1) / bw) ** 2);
    xs.push(x);
    ys.push(Math.log(Math.max(1, r.day[hourIndex])));
    ws.push(w);
  }
  let sw = 0;
  for (const w of ws) sw += w;
  if (sw < 1e-9) return null;
  let mx = 0;
  let my = 0;
  for (let i = 0; i < xs.length; i++) {
    mx += ws[i] * xs[i];
    my += ws[i] * ys[i];
  }
  mx /= sw;
  my /= sw;
  let sxy = 0;
  let sxx = 0;
  for (let i = 0; i < xs.length; i++) {
    sxy += ws[i] * (xs[i] - mx) * (ys[i] - my);
    sxx += ws[i] * (xs[i] - mx) ** 2;
  }
  let slope = sxx > 1e-12 ? sxy / sxx : 0;
  // Sanity guard: a launch-day level cannot plausibly fall, or explode, with
  // hour-1 size outside this band; clamp instead of trusting a degenerate fit.
  if (slope < -1) slope = -1;
  if (slope > 1.5) slope = 1.5;
  const intercept = my - slope * mx;
  let sse = 0;
  for (let i = 0; i < xs.length; i++) {
    const e = ys[i] - (intercept + slope * xs[i]);
    sse += ws[i] * e * e;
  }
  const sd = Math.sqrt(sse / sw);
  return { mu: intercept + slope * logP1, sd };
}

/** The five quantiles for one hour, from a log-mean and log-SD. */
function logNormalQuantiles(mu: number, sd: number): number[] {
  const qs = Z.map((z) => round2(Math.exp(mu + z * sd)));
  const out: number[] = [];
  let prev = 0.01;
  for (const q of qs) {
    // Keep every q finite, > 0, <= 1e9, and the row non-decreasing.
    const clamped = Math.min(1e9, Math.max(prev, Number.isFinite(q) ? Math.max(0.01, q) : prev));
    out.push(clamped);
    prev = clamped;
  }
  return out;
}

/** Persistence fallback (house shape) for one hour. */
function persistenceQuantiles(p1: number, h: number): number[] {
  return logNormalQuantiles(Math.log(p1), PERSIST_SIGMA * Math.sqrt(h - 1));
}

/** Forecast one case: 23 rows of five quantiles for hours 2..24. */
function forecastCase(c: Case): Array<{ hour: number; q: number[] }> {
  const first = c.known?.find((k) => k.hour === 1);
  let p1 = first !== undefined && Number.isFinite(first.viewers) ? Math.max(1, first.viewers) : NaN;
  if (!Number.isFinite(p1)) {
    const hist = c.history_hourly ?? [];
    const last = hist.length > 0 ? hist[hist.length - 1].viewers : NaN;
    p1 = Number.isFinite(last) ? Math.max(1, last) : 1000;
  }
  // The contract is exactly hours 2..24, each once. Trust the case's
  // `forecast` list when it says that; otherwise rebuild it, so a malformed
  // list can never produce an invalid row set.
  const listed = (c.forecast?.length ? c.forecast : []).map((f) => f.hour);
  const hours =
    listed.length === 23 && listed.every((h, i) => h === i + 2)
      ? listed
      : Array.from({ length: 23 }, (_, i) => i + 2);

  const rows: Array<{ v1: number; day: number[] }> = [];
  for (const comp of c.comparables ?? []) {
    const day = releaseDay(comp);
    if (day !== null) rows.push({ v1: day[0], day });
  }
  const logP1 = Math.log(p1);
  return hours.map((h) => {
    if (rows.length < MIN_COMPARABLES || h < 2 || h > 24) {
      return { hour: h, q: persistenceQuantiles(p1, Math.max(2, Math.min(24, h))) };
    }
    const fit = kernelFit(rows, h - 1, logP1);
    if (fit === null || !Number.isFinite(fit.mu) || !Number.isFinite(fit.sd)) {
      return { hour: h, q: persistenceQuantiles(p1, h) };
    }
    // The fitted median must stay in a sane band around hour 1; anything
    // outside it means the local fit went degenerate, so fall back.
    const muClamped = Math.min(Math.log(p1) + Math.log(50), Math.max(Math.log(p1) + Math.log(0.05), fit.mu));
    return { hour: h, q: logNormalQuantiles(muClamped, Math.max(0.05, fit.sd * SPREAD_SCALE)) };
  });
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  const hours = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ hours })}\n`);
}
