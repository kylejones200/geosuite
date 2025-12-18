import { initMap } from './map.js';
import { initChat } from './chat.js';
import { initOperators } from './operators.js';

window.addEventListener('DOMContentLoaded', () => {
  initMap();
  initChat();
  initOperators();

  // Draggable resizer to adjust map/side split
  const app = document.getElementById('app');
  const resizer = document.getElementById('resizer');
  if (app && resizer) {
    let dragging = false;
    const minSide = 300; // px
    const maxSide = 800; // px

    // Restore saved width on load
    const saved = localStorage.getItem('sideWidth');
    if (saved) {
      const val = Math.max(minSide, Math.min(maxSide, parseInt(saved, 10) || 0));
      if (val) document.documentElement.style.setProperty('--side-width', `${val}px`);
    }
    const onMove = (e) => {
      if (!dragging) return;
      const x = e.touches ? e.touches[0].clientX : e.clientX;
      const rect = app.getBoundingClientRect();
      const total = rect.width;
      let side = total - (x - rect.left);
      side = Math.max(minSide, Math.min(maxSide, side));
      document.documentElement.style.setProperty('--side-width', `${side}px`);
    };
    const stop = () => {
      if (dragging) {
        // persist current width
        const cs = getComputedStyle(document.documentElement).getPropertyValue('--side-width').trim();
        const px = parseInt(cs.replace('px',''), 10);
        if (px) localStorage.setItem('sideWidth', String(px));
      }
      dragging = false;
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('touchmove', onMove);
      window.removeEventListener('mouseup', stop);
      window.removeEventListener('touchend', stop);
    };
    const start = (e) => { dragging = true; window.addEventListener('mousemove', onMove); window.addEventListener('touchmove', onMove, { passive: false }); window.addEventListener('mouseup', stop); window.addEventListener('touchend', stop); e.preventDefault(); };
    resizer.addEventListener('mousedown', start);
    resizer.addEventListener('touchstart', start, { passive: false });
  }
});
