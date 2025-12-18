// UI helpers: toast + well detail rendering
import { fitHyperbolic, hyperbolicCumulative, hyperbolicForecast } from './dca.js';
import { renderChart, renderCuts } from './charting.js';

export function toast(msg, ms = 1800) {
  const t = document.getElementById('toast');
  if (!t) return;
  t.textContent = msg;
  t.style.display = 'block';
  setTimeout(() => (t.style.display = 'none'), ms);
}

export function showWellDetail(props) {
  const p = props || {};
  const title = document.getElementById('wellTitle');
  const meta = document.getElementById('wellMeta');
  if (title) title.textContent = p.api || '—';
  if (meta) meta.textContent = `${p.operator || '—'} • ${p.formation || '—'} • First Prod ${p.first_prod || '—'} • Cum ${Number(p.cum_oil || 0).toLocaleString()} bbl`;

  let times = [];
  if (Array.isArray(p.timeseries)) {
    times = p.timeseries;
  } else if (typeof p.timeseries === 'string') {
    try { times = JSON.parse(p.timeseries) || []; } catch (_) { times = []; }
  }
  if (!times.length) { toast('No time-series data for this well'); return; }

  const fit = fitHyperbolic(times);
  const kqi = document.getElementById('kqi');
  const kdi = document.getElementById('kdi');
  const kb = document.getElementById('kb');
  const keur = document.getElementById('keur');
  if (kqi) kqi.textContent = Math.round(fit.qi);
  if (kdi) kdi.textContent = fit.Di.toFixed(4);
  if (kb) kb.textContent = fit.b.toFixed(2);

  // Forecast to an economic limit rate (default 30 bopd)
  const tStart = times.length ? times[times.length - 1].t : 0;
  const econLimit = 30; // bopd
  const forecast = hyperbolicForecast(fit, tStart, econLimit, 15, 365 * 30);

  // Compute EUR: observed cum from server + forecast cumulative beyond tStart
  const npStart = hyperbolicCumulative(fit, tStart);
  const npEnd = hyperbolicCumulative(fit, forecast.tEnd);
  const forecastAdd = Math.max(0, npEnd - npStart);
  const observedCum = Number(p.cum_oil || 0);
  const eur = observedCum + forecastAdd;
  if (keur) keur.textContent = (eur / 1000).toFixed(1);

  renderChart(times, fit, forecast);
  // Render cut charts using server-provided water/gas rate arrays aligned with timeseries
  const waterRates = Array.isArray(p.water_rates) ? p.water_rates : [];
  const gasRates = Array.isArray(p.gas_rates) ? p.gas_rates : [];
  renderCuts(times, waterRates, gasRates);
}
