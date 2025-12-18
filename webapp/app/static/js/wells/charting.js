// Chart rendering utilities using Chart.js global
import { hyperbolicCurve } from './dca.js';

let chart;
let waterCutChart;
let gasCutChart;

export function renderChart(times, model, forecast = null) {
  const canvas = document.getElementById('prodChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  const labels = times.map(d => d.t);
  const prod = times.map(d => d.q);
  const tmax = Math.max(...labels, 0);
  const dca = hyperbolicCurve(model, tmax);

  // Build XY points so we can use a linear x-axis with nice numeric ticks
  const oilPoints = times.map(d => ({ x: d.t, y: d.q }));
  const dcaPoints = dca.map(d => ({ x: d.t, y: d.q }));
  let forecastPoints = [];
  if (forecast && Array.isArray(forecast.series) && forecast.series.length) {
    const fSeries = forecast.series;
    const tLast = labels.length ? labels[labels.length - 1] : 0;
    const ext = fSeries.filter(p => p.t > tLast);
    if (ext.length) forecastPoints = ext.map(p => ({ x: p.t, y: p.q }));
  }

  const niceMax = (n) => {
    const step = 365; // whole years in days
    return Math.ceil(n / step) * step;
  };
  const xMax = niceMax(Math.max(tmax, forecastPoints.length ? forecastPoints[forecastPoints.length - 1].x : 0));

  if (chart) chart.destroy();
  chart = new Chart(ctx, {
    type: 'line',
    data: {
      datasets: [
        { label: 'Oil rate (bopd)', data: oilPoints, parsing: { xAxisKey: 'x', yAxisKey: 'y' }, borderWidth: 2, pointRadius: 2, tension: .2, borderColor: '#60a5fa', pointBackgroundColor: '#fff', pointBorderColor: '#000', pointStyle: 'line', showLine: true },
        { label: 'Hyperbolic DCA', data: dcaPoints, parsing: { xAxisKey: 'x', yAxisKey: 'y' }, borderWidth: 3, pointRadius: 0, borderColor: '#34d399', pointStyle: 'line', showLine: true },
        ...(forecastPoints.length ? [{ label: 'Forecast', data: forecastPoints, parsing: { xAxisKey: 'x', yAxisKey: 'y' }, borderWidth: 3, pointRadius: 0, borderDash: [6,4], borderColor: '#f87171', pointStyle: 'line', showLine: true }] : [])
      ]
    },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        // Use line samples in legend instead of filled/outlined boxes
        legend: { labels: { color: '#cbd5e1', usePointStyle: true } },
        // Darker tooltip text for better readability
        tooltip: {
          mode: 'index',
          intersect: false,
          titleColor: '#94a3b8', // darker than labels
          bodyColor: '#94a3b8',
          footerColor: '#94a3b8',
          backgroundColor: 'rgba(17,24,39,0.9)', // dark slate background
          borderColor: 'rgba(148,163,184,0.4)',
          borderWidth: 1
        }
      },
      scales: {
        x: { type: 'linear', title: { display: true, text: 'Years', color: '#9aa0a6' },
             ticks: { color: '#9aa0a6', stepSize: 365, maxRotation: 0, minRotation: 0, autoSkip: true, callback: (v) => {
               // Convert days to whole years (no decimals, no unit letters)
               return Math.round(v / 365);
             } },
             grid: { color: 'rgba(255,255,255,0.04)' }, suggestedMax: xMax, beginAtZero: true },
        y: { title: { display: true, text: 'bopd', color: '#9aa0a6' }, ticks: { color: '#9aa0a6' }, grid: { color: 'rgba(255,255,255,0.04)' } }
      }
    }
  });
}

// Render Oil/Water cut and Oil/Gas cut charts
export function renderCuts(times, waterRates = [], gasRates = []) {
  const oilRates = Array.isArray(times) ? times.map(d => d.q || 0) : [];
  const labels = Array.isArray(times) ? times.map(d => d.t) : [];

  // Compute cuts as percentages (oil share of oil+water and oil+gas)
  const ow = labels.map((_, i) => {
    const o = oilRates[i] || 0;
    const w = waterRates[i] || 0;
    const denom = o + w;
    return denom > 0 ? (o / denom) * 100 : null;
  });
  const og = labels.map((_, i) => {
    const o = oilRates[i] || 0;
    const g = gasRates[i] || 0;
    const denom = o + g;
    return denom > 0 ? (o / denom) * 100 : null;
  });

  const wCanvas = document.getElementById('waterCutChart');
  const gCanvas = document.getElementById('gasCutChart');
  if (!wCanvas || !gCanvas) return;

  if (waterCutChart) waterCutChart.destroy();
  waterCutChart = new Chart(wCanvas.getContext('2d'), {
    type: 'line',
    data: { labels, datasets: [{ label: 'Oil/Water Cut (%)', data: ow, borderColor: '#93c5fd', borderWidth: 2, pointRadius: 0 }] },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: '#cbd5e1', usePointStyle: true } },
        tooltip: { titleColor: '#94a3b8', bodyColor: '#94a3b8', backgroundColor: 'rgba(17,24,39,0.9)', borderColor: 'rgba(148,163,184,0.4)', borderWidth: 1 }
      },
      scales: {
        x: { ticks: { color: '#9aa0a6' }, grid: { color: 'rgba(255,255,255,0.04)' } },
        y: { ticks: { color: '#9aa0a6' }, grid: { color: 'rgba(255,255,255,0.04)' }, min: 0, max: 100 }
      }
    }
  });

  if (gasCutChart) gasCutChart.destroy();
  gasCutChart = new Chart(gCanvas.getContext('2d'), {
    type: 'line',
    data: { labels, datasets: [{ label: 'Oil/Gas Cut (%)', data: og, borderColor: '#fbbf24', borderWidth: 2, pointRadius: 0 }] },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: '#cbd5e1', usePointStyle: true } },
        tooltip: { titleColor: '#94a3b8', bodyColor: '#94a3b8', backgroundColor: 'rgba(17,24,39,0.9)', borderColor: 'rgba(148,163,184,0.4)', borderWidth: 1 }
      },
      scales: {
        x: { ticks: { color: '#9aa0a6' }, grid: { color: 'rgba(255,255,255,0.04)' } },
        y: { ticks: { color: '#9aa0a6' }, grid: { color: 'rgba(255,255,255,0.04)' }, min: 0, max: 100 }
      }
    }
  });
}
