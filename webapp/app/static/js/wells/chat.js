// Simple chat wiring; replace endpoint in askGenie with your backend
import { toast } from './ui.js';

export const GENIE_ENDPOINT = '/api/genie';

function addMsg(txt, who) {
  const chatlog = document.getElementById('chatlog');
  if (!chatlog) return;
  const div = document.createElement('div');
  div.className = 'msg ' + (who === 'u' ? 'u' : 'b');
  div.textContent = txt;
  chatlog.appendChild(div);
  chatlog.scrollTop = chatlog.scrollHeight;
}

async function askGenie(text) {
  return fetch(GENIE_ENDPOINT, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question: text })
  }).then(r => r.json());
}

export function initChat() {
  const askBtn = document.getElementById('ask');
  const input = document.getElementById('q');
  if (!askBtn || !input) return;

  askBtn.addEventListener('click', async () => {
    const text = input.value.trim();
    if (!text) return;
    addMsg(text, 'u');
    input.value = '';
    try {
      const res = await askGenie(text);
      addMsg(res.text || 'No response', 'b');
    } catch (e) {
      console.error(e);
      toast('Chat request failed');
    }
  });

  input.addEventListener('keydown', e => {
    if (e.key === 'Enter') askBtn.click();
  });
}
