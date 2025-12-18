// Map initialization using MapLibre (global `maplibregl`) and Mapbox GL Draw (global `MapboxDraw`)
import { showWellDetail } from './ui.js';
import { setOperatorsData } from './operators.js';
import { fetchWellsData } from './data_api.js';

let popup;

export function initMap() {
  const map = new maplibregl.Map({
    container: 'map',
    style: 'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json',
    center: [-102.5, 47.5],
    zoom: 6.2,
    attributionControl: false
  });
  map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'top-right');

  popup = new maplibregl.Popup({ closeButton: false, closeOnClick: false });

  map.on('load', () => {
    // Create empty source; will be populated from API
    map.addSource('wells', { type: 'geojson', data: { type: 'FeatureCollection', features: [] }, promoteId: 'id' });
    
    // Load wells data from Flask API
    loadWellsFromAPI(map);
    map.addLayer({
      id: 'wells',
      type: 'circle',
      source: 'wells',
      paint: {
        'circle-color': [
          'case',
          ['==', ['get', 'formation'], 'Bakken'], '#60a5fa',
          ['==', ['get', 'formation'], 'Three Forks'], '#34d399',
          '#fbbf24'
        ],
        'circle-radius': ['interpolate', ['linear'], ['zoom'], 4, 3, 8, 6, 12, 8],
        'circle-opacity': 0.85,
        'circle-stroke-color': '#0b0c10',
        'circle-stroke-width': 1
      }
    });

    // Fetch wells from backend
    fetch('/api/wells')
      .then(r => r.json())
      .then(geojson => {
        const src = map.getSource('wells');
        if (src) src.setData(geojson);
      })
      .catch(err => {
        console.error('Failed to load wells', err);
      });

    map.on('mousemove', 'wells', e => {
      map.getCanvas().style.cursor = 'pointer';
      const p = e.features[0].properties;
      const html = `
        <div style="font-size:12px">
          <b>${p.api}</b><br/>
          ${p.operator}<br/>
          ${p.formation}
        </div>
      `;
      popup.setLngLat(e.lngLat).setHTML(html).addTo(map);
    });
    map.on('mouseleave', 'wells', () => { map.getCanvas().style.cursor = ''; popup.remove(); });
    map.on('click', 'wells', async e => {
      const feat = e.features[0];
      const uwi = feat.id || (feat.properties && feat.properties.api);
      if (!uwi) return;
      try {
        const res = await fetch(`/api/wells/${encodeURIComponent(uwi)}`);
        const detail = await res.json();
        showWellDetail(detail);
      } catch (err) {
        console.error('Failed to load well detail', err);
      }
    });

    // --- Polygon selection using Mapbox GL Draw ---
    try {
      // Only attach if Draw is available
      if (window.MapboxDraw) {
        const draw = new window.MapboxDraw({
          displayControlsDefault: false,
          controls: { polygon: true, trash: true },
          defaultMode: 'draw_polygon'
        });
        map.addControl(draw, 'top-left');

        async function updateSelection() {
          const fc = draw.getAll();
          const status = document.getElementById('opStatus');
          const sinceVal = (document.getElementById('sinceYear') && document.getElementById('sinceYear').value) || '';
          // If there is a polygon, post to backend to aggregate
          const poly = fc.features.find(f => f.geometry && f.geometry.type === 'Polygon');
          if (poly) {
            try {
              const resp = await fetch('/api/operators/within', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ geometry: poly.geometry, since: sinceVal ? Number(sinceVal) : undefined })
              });
              const json = await resp.json();
              setOperatorsData((json && json.operators) || []);
              if (status) status.textContent = `Selection: ${json.wells || 0} wells in polygon`;
            } catch (e) {
              console.error('Failed selection aggregation', e);
            }
          } else {
            // No polygon → reset to all operators
            try {
              const resp = await fetch(`/api/operators${sinceVal ? `?since=${encodeURIComponent(sinceVal)}` : ''}`);
              const json = await resp.json();
              setOperatorsData((json && json.operators) || []);
              if (status) status.textContent = 'Selection: All wells';
            } catch (e) {
              console.error('Failed to reload all operators', e);
            }
          }
        }

        map.on('draw.create', updateSelection);
        map.on('draw.update', updateSelection);
        map.on('draw.delete', updateSelection);

        const clearBtn = document.getElementById('clearSelection');
        if (clearBtn) {
          clearBtn.addEventListener('click', () => {
            draw.deleteAll();
            updateSelection();
  });
}

// Load wells data from Flask API
async function loadWellsFromAPI(map) {
  try {
    console.log('Loading wells from API...');
    const wellsData = await fetchWellsData();
    
    if (wellsData && wellsData.features) {
      console.log(`Loaded ${wellsData.features.length} wells from API`);
      map.getSource('wells').setData(wellsData);
    } else {
      console.error('No wells data received from API');
    }
  } catch (error) {
    console.error('Error loading wells from API:', error);
  }
}

        // Expose for other modules (operators.js applySince)
        window._updateOperatorSelection = updateSelection;
      }
    } catch (e) {
      console.warn('Mapbox Draw not available', e);
    }
  });

  return map;
}
