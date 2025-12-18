// Decline Curve Analysis (hyperbolic) utilities
// Exported API:
// - fitHyperbolic(times)
// - hyperbolicCurve(params, tmax, dt)

// times: Array<{t:number, q:number}>
export function fitHyperbolic(times) {
  const t = times.map(d => d.t);
  const q = times.map(d => d.q);
  const qmax = Math.max(...q);
  const qmin = Math.max(1, Math.min(...q));
  let best = { qi: qmax, Di: 0.01, b: 0.8, sse: Infinity };

  const bGrid = [0.3, 0.5, 0.7, 1.0, 1.3, 1.5];
  const DiGrid = [0.002, 0.004, 0.006, 0.01, 0.015, 0.02, 0.03];

  for (const b of bGrid) {
    for (const Di of DiGrid) {
      const f = t.map(tt => Math.pow(1 + b * Di * tt, -1 / b));
      const num = q.reduce((acc, qi_i, idx) => acc + qi_i * f[idx], 0);
      const den = f.reduce((acc, fi) => acc + fi * fi, 0) || 1e-9;
      const qi = Math.max(qmax * 0.7, Math.min(qmax * 1.5, num / den));
      const sse = q.reduce((acc, qi_i, idx) => {
        const qhat = qi * f[idx];
        return acc + Math.pow(qi_i - qhat, 2);
      }, 0);
      if (sse < best.sse) best = { qi, Di, b, sse };
    }
  }

  const steps = [0.5, 0.2, 0.1];
  for (const step of steps) {
    for (const _ of [0, 1, 2, 3]) {
      const cands = [];
      for (const dDi of [-1, 0, 1]) {
        for (const db of [-1, 0, 1]) {
          const Di = Math.max(1e-4, best.Di * Math.pow(1.5, dDi * step));
          const b = Math.max(0.1, Math.min(2.0, best.b * Math.pow(1.5, db * step)));
          const f = t.map(tt => Math.pow(1 + b * Di * tt, -1 / b));
          const num = q.reduce((acc, qi_i, idx) => acc + qi_i * f[idx], 0);
          const den = f.reduce((acc, fi) => acc + fi * fi, 0) || 1e-9;
          const qi = Math.max(qmax * 0.5, Math.min(qmax * 2.0, num / den));
          const sse = q.reduce((acc, qi_i, idx) => acc + Math.pow(qi_i - qi * f[idx], 2), 0);
          cands.push({ qi, Di, b, sse });
        }
      }
      cands.sort((a, b) => a.sse - b.sse);
      if (cands[0].sse < best.sse) best = cands[0];
    }
  }
  return best;
}

export function hyperbolicCurve(params, tmax, dt = 15) {
  const { qi, Di, b } = params;
  const out = [];
  for (let t = 0; t <= tmax; t += dt) {
    const q = qi * Math.pow(1 + b * Di * t, -1 / b);
    out.push({ t, q });
  }
  return out;
}

// Cumulative production for hyperbolic decline Np(t) in barrels
// Handles b != 1 and b == 1 (harmonic)
export function hyperbolicCumulative(params, t) {
  const { qi, Di, b } = params;
  if (t <= 0) return 0;
  if (Math.abs(b - 1) < 1e-6) {
    // Harmonic: Np = (qi/Di) * ln(1 + Di t)
    return (qi / Di) * Math.log(1 + Di * t);
  }
  // General hyperbolic: Np = qi/((1-b)Di) * [1 - (1 + b Di t)^{(b-1)/b}]
  const term = Math.pow(1 + b * Di * t, (b - 1) / b);
  return (qi / ((1 - b) * Di)) * (1 - term);
}

// Compute time (days) when rate declines to qLimit bopd, starting from t0
export function timeToRate(params, qLimit, t0 = 0) {
  const { qi, Di, b } = params;
  const q0 = qi * Math.pow(1 + b * Di * t0, -1 / b);
  if (qLimit >= q0) return t0;
  // Solve for t where q(t) = qLimit: (1 + b Di t) = (qi/q)^b
  const rhs = Math.pow(qi / qLimit, b);
  const t = (rhs - 1) / (b * Di);
  return Math.max(t, t0);
}

// Build forecast curve starting at tStart until q falls to qLimit (or maxDays)
export function hyperbolicForecast(params, tStart, qLimit = 30, dt = 15, maxDays = 365 * 30) {
  const out = [];
  const tEnd = Math.min(timeToRate(params, qLimit, tStart), tStart + maxDays);
  for (let t = tStart; t <= tEnd; t += dt) {
    const q = params.qi * Math.pow(1 + params.b * params.Di * t, -1 / params.b);
    out.push({ t, q });
  }
  return { series: out, tEnd };
}
