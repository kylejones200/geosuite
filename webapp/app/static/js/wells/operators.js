// Operators leaderboard: fetch and render operator performance
import { fetchOperatorsData } from './data_api.js';

function fmtInt(n) { return Number(n || 0).toLocaleString(); }
function fmtBbl(n) { return Number(n || 0).toLocaleString(undefined, { maximumFractionDigits: 0 }); }
function fmtRate(n) { return Number(n || 0).toLocaleString(undefined, { maximumFractionDigits: 0 }); }

function renderTable(rows, tableEl) {
  const tbody = tableEl.querySelector('tbody');
  tbody.innerHTML = '';
  rows.forEach((r, idx) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="c">${idx + 1}</td>
      <td>${r.operator}</td>
      <td class="c">${fmtInt(r.wells)}</td>
      <td class="c">${fmtRate(r.avg_qi)}</td>
      <td class="r">${fmtBbl(r.total_cum_oil)}</td>
    `;
    tbody.appendChild(tr);
  });
}

let _data = [];
let _sortKey = 'total_cum_oil';
let _sortDir = 'desc';
let _barChart = null;

function getSort() { return { sortKey: _sortKey, sortDir: _sortDir }; }

function applySortAndRender(table) {
  const rows = [..._data];
  rows.sort((a, b) => {
    const va = a[_sortKey];
    const vb = b[_sortKey];
    if (typeof va === 'string' && typeof vb === 'string') {
      return _sortDir === 'asc' ? va.localeCompare(vb) : vb.localeCompare(va);
    }
    return _sortDir === 'asc' ? (va - vb) : (vb - va);
  });
  renderTable(rows, document.getElementById('opTable'));
  const tableEl = document.getElementById('opTable');
  tableEl.querySelectorAll('th').forEach(th => th.classList.remove('sort-asc', 'sort-desc'));
  const th = tableEl.querySelector(`th[data-key="${_sortKey}"]`);
  if (th) th.classList.add(_sortDir === 'asc' ? 'sort-asc' : 'sort-desc');

  // Render bar chart (Top 10 by wells)
  try {
    const barEl = document.getElementById('opBar');
    if (barEl) {
      const top = rows.slice(0, 10);
      const labels = top.map(r => r.operator);
      const wells = top.map(r => r.wells);
      if (_barChart) _barChart.destroy();
      _barChart = new Chart(barEl.getContext('2d'), {
        type: 'bar',
        data: { labels, datasets: [{ label: 'Wells', data: wells, backgroundColor: '#60a5fa' }] },
        options: {
          animation: false,
          plugins: { legend: { labels: { color: '#cbd5e1' } }, tooltip: { titleColor: '#94a3b8', bodyColor: '#94a3b8', backgroundColor: 'rgba(17,24,39,0.9)', borderColor: 'rgba(148,163,184,0.4)', borderWidth: 1 } },
          scales: { x: { ticks: { color: '#9aa0a6' } }, y: { ticks: { color: '#9aa0a6' }, beginAtZero: true } }
        }
      });
    }
  } catch (e) { /* noop */ }
}

export function setOperatorsData(list) {
  _data = Array.isArray(list) ? list : [];
  applySortAndRender(document.getElementById('opTable'));
}

export async function initOperators() {
  const table = document.getElementById('opTable');
  if (!table) return;

  try {
    const since = (document.getElementById('sinceYear') && document.getElementById('sinceYear').value) || '';
    const res = await fetch(`/api/operators${since ? `?since=${encodeURIComponent(since)}` : ''}`);
    const json = await res.json();
    _data = (json && json.operators) || [];
    applySortAndRender(table);
  } catch (err) {
    console.error('Failed to load operators', err);
  }

  // Sorting handlers
  table.querySelectorAll('th').forEach(th => {
    th.addEventListener('click', () => {
      const key = th.getAttribute('data-key');
      if (!key) return;
      if (_sortKey === key) {
        _sortDir = _sortDir === 'asc' ? 'desc' : 'asc';
      } else {
        _sortKey = key;
        _sortDir = key === 'operator' ? 'asc' : 'desc';
      }
      applySortAndRender(table);
    });
  });

  // Apply since-year filter
  const applyBtn = document.getElementById('applySince');
  if (applyBtn) {
    applyBtn.addEventListener('click', async () => {
      if (window._updateOperatorSelection) {
        // Re-run selection flow (it will read the since value when posting)
        window._updateOperatorSelection();
      } else {
        // Fallback: reload all operators with since
        try {
          const sinceVal = (document.getElementById('sinceYear') && document.getElementById('sinceYear').value) || '';
          const resp = await fetch(`/api/operators${sinceVal ? `?since=${encodeURIComponent(sinceVal)}` : ''}`);
          const json = await resp.json();
          setOperatorsData((json && json.operators) || []);
        } catch (e) { console.error('Failed to apply since filter', e); }
      }
    });
  }

  // CSV download of current, sorted table
  const dlBtn = document.getElementById('downloadCSV');
  if (dlBtn) {
    dlBtn.addEventListener('click', () => {
      // Sort current data same as display
      const rows = [..._data].sort((a, b) => {
        const va = a[_sortKey]; const vb = b[_sortKey];
        if (typeof va === 'string' && typeof vb === 'string') return _sortDir === 'asc' ? va.localeCompare(vb) : vb.localeCompare(va);
        return _sortDir === 'asc' ? (va - vb) : (vb - va);
      });
      const header = ['Operator', 'Wells', 'Avg_qi_bopd', 'Total_Cum_bbl'];
      const lines = [header.join(',')].concat(rows.map(r => [
        '"' + String(r.operator).replace(/"/g,'""') + '"',
        r.wells,
        Math.round(r.avg_qi),
        Math.round(r.total_cum_oil)
      ].join(',')));
      const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8;' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = 'operator_performance.csv';
      document.body.appendChild(a); a.click(); document.body.removeChild(a);
      URL.revokeObjectURL(url);
    });
  }
}
