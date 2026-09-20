"""Embedded Interactive Web Chat UI for System One & System Two Engines."""


def get_chat_html() -> str:
    """Return standalone, responsive HTML/CSS/JS chat interface with streaming and code highlighting."""
    return r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>System One Engine — Live System 1 + System 2 Chat</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-dark: #090d16;
      --bg-card: rgba(18, 24, 38, 0.85);
      --bg-bubble-user: #4f46e5;
      --bg-bubble-bot: rgba(30, 41, 59, 0.9);
      --border-color: rgba(255, 255, 255, 0.08);
      --text-primary: #f8fafc;
      --text-secondary: #94a3b8;
      --text-muted: #64748b;
      --accent-purple: #8b5cf6;
      --accent-cyan: #06b6d4;
      --accent-emerald: #10b981;
      --accent-rose: #f43f5e;
      --accent-amber: #f59e0b;
      --glow-purple: rgba(139, 92, 246, 0.25);
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      font-family: 'Inter', sans-serif;
      background-color: var(--bg-dark);
      background-image: 
        radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
        radial-gradient(at 100% 100%, rgba(139, 92, 246, 0.1) 0px, transparent 50%);
      color: var(--text-primary);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 1rem;
    }

    .chat-container {
      width: 100%;
      max-width: 950px;
      height: 94vh;
      display: flex;
      flex-direction: column;
      background: var(--bg-card);
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
      border: 1px solid var(--border-color);
      border-radius: 20px;
      box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.6), 0 0 40px var(--glow-purple);
      overflow: hidden;
    }

    /* Header */
    header {
      padding: 1.1rem 1.75rem;
      border-bottom: 1px solid var(--border-color);
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: rgba(15, 23, 42, 0.6);
    }

    .brand-title {
      display: flex;
      align-items: center;
      gap: 0.85rem;
    }

    .brand-logo {
      width: 38px;
      height: 38px;
      background: linear-gradient(135deg, #6366f1, #8b5cf6);
      border-radius: 10px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 700;
      color: #ffffff;
      font-size: 1.1rem;
      box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
    }

    .brand-text h1 {
      font-size: 1.1rem;
      font-weight: 600;
      letter-spacing: -0.01em;
    }

    .brand-text p {
      font-size: 0.75rem;
      color: var(--text-secondary);
    }

    .header-badges {
      display: flex;
      gap: 0.5rem;
      align-items: center;
    }

    .badge-status {
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      padding: 0.35rem 0.75rem;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 500;
      background: rgba(16, 185, 129, 0.12);
      color: var(--accent-emerald);
      border: 1px solid rgba(16, 185, 129, 0.25);
    }

    .badge-status::before {
      content: '';
      width: 6px;
      height: 6px;
      background: var(--accent-emerald);
      border-radius: 50%;
      box-shadow: 0 0 8px var(--accent-emerald);
    }

    .badge-s2 {
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      padding: 0.35rem 0.75rem;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 500;
      background: rgba(139, 92, 246, 0.12);
      color: var(--accent-purple);
      border: 1px solid rgba(139, 92, 246, 0.25);
    }

    /* Messages Area */
    #messages-box {
      flex: 1;
      padding: 1.5rem;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
      scroll-behavior: smooth;
    }

    #messages-box::-webkit-scrollbar {
      width: 6px;
    }
    #messages-box::-webkit-scrollbar-thumb {
      background: rgba(255, 255, 255, 0.1);
      border-radius: 3px;
    }

    .message-row {
      display: flex;
      flex-direction: column;
      max-width: 88%;
      animation: fadeIn 0.25s ease-out;
    }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(8px); }
      to { opacity: 1; transform: translateY(0); }
    }

    .message-row.user {
      align-self: flex-end;
      align-items: flex-end;
    }

    .message-row.bot {
      align-self: flex-start;
      align-items: flex-start;
      width: 100%;
    }

    .bubble {
      padding: 0.95rem 1.25rem;
      border-radius: 16px;
      font-size: 0.93rem;
      line-height: 1.6;
      word-break: break-word;
    }

    .bubble.user {
      background: linear-gradient(135deg, #6366f1, #4f46e5);
      color: #ffffff;
      border-bottom-right-radius: 4px;
      box-shadow: 0 4px 15px rgba(99, 102, 241, 0.25);
    }

    .bubble.bot {
      background: var(--bg-bubble-bot);
      border: 1px solid var(--border-color);
      color: var(--text-primary);
      border-bottom-left-radius: 4px;
      width: 100%;
    }

    /* Code block styling */
    pre.code-block {
      background: #0d1117;
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 10px;
      margin: 0.75rem 0;
      overflow-x: auto;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.85rem;
      position: relative;
    }

    .code-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: rgba(255, 255, 255, 0.04);
      padding: 0.35rem 0.8rem;
      border-bottom: 1px solid rgba(255, 255, 255, 0.06);
      font-size: 0.72rem;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }

    .copy-btn {
      background: transparent;
      border: 1px solid rgba(255, 255, 255, 0.15);
      color: var(--text-secondary);
      border-radius: 6px;
      padding: 0.2rem 0.5rem;
      font-size: 0.7rem;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .copy-btn:hover {
      background: rgba(255, 255, 255, 0.1);
      color: #fff;
    }

    pre.code-block code {
      display: block;
      padding: 0.85rem 1rem;
      color: #e2e8f0;
      line-height: 1.5;
    }

    code.inline-code {
      font-family: 'JetBrains Mono', monospace;
      background: rgba(255, 255, 255, 0.08);
      padding: 0.15rem 0.35rem;
      border-radius: 4px;
      font-size: 0.85em;
      color: var(--accent-cyan);
    }

    /* System One Telemetry Card */
    .telemetry-card {
      margin-top: 0.6rem;
      padding: 0.75rem 1rem;
      background: rgba(15, 23, 42, 0.75);
      border: 1px solid rgba(255, 255, 255, 0.06);
      border-radius: 12px;
      font-size: 0.8rem;
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
      width: 100%;
    }

    .telemetry-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid rgba(255, 255, 255, 0.05);
      padding-bottom: 0.4rem;
    }

    .telemetry-title {
      font-size: 0.72rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-secondary);
      display: flex;
      align-items: center;
      gap: 0.4rem;
    }

    .telemetry-latency {
      font-family: 'JetBrains Mono', monospace;
      color: var(--accent-cyan);
      font-weight: 600;
      font-size: 0.75rem;
    }

    .telemetry-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 0.6rem;
    }

    .tag-item {
      display: flex;
      flex-direction: column;
      gap: 0.15rem;
    }

    .tag-label {
      font-size: 0.7rem;
      color: var(--text-muted);
    }

    .tag-val {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.8rem;
      font-weight: 500;
    }

    .pill-urgent {
      color: var(--accent-rose);
      font-weight: 600;
    }
    .pill-normal {
      color: var(--accent-emerald);
    }

    /* Suggestion Chips */
    .chips-tray {
      padding: 0.5rem 1.25rem;
      display: flex;
      gap: 0.5rem;
      overflow-x: auto;
      border-top: 1px solid var(--border-color);
      background: rgba(15, 23, 42, 0.3);
    }

    .chips-tray::-webkit-scrollbar {
      display: none;
    }

    .chip-btn {
      white-space: nowrap;
      padding: 0.45rem 0.85rem;
      border-radius: 9999px;
      font-size: 0.78rem;
      background: rgba(255, 255, 255, 0.04);
      color: var(--text-secondary);
      border: 1px solid var(--border-color);
      cursor: pointer;
      transition: all 0.15s ease;
    }

    .chip-btn:hover {
      background: rgba(99, 102, 241, 0.15);
      border-color: rgba(99, 102, 241, 0.4);
      color: var(--text-primary);
    }

    /* Input Footer */
    footer {
      padding: 1rem 1.25rem;
      background: rgba(15, 23, 42, 0.5);
      border-top: 1px solid var(--border-color);
      display: flex;
      gap: 0.75rem;
      align-items: center;
    }

    #chat-input {
      flex: 1;
      padding: 0.85rem 1.15rem;
      border-radius: 12px;
      background: rgba(15, 23, 42, 0.75);
      border: 1px solid var(--border-color);
      color: var(--text-primary);
      font-family: inherit;
      font-size: 0.92rem;
      outline: none;
      transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }

    #chat-input:focus {
      border-color: #6366f1;
      box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
    }

    #send-btn {
      padding: 0.85rem 1.4rem;
      border-radius: 12px;
      background: linear-gradient(135deg, #6366f1, #4f46e5);
      color: #ffffff;
      font-weight: 600;
      font-size: 0.9rem;
      border: none;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      transition: transform 0.15s ease, box-shadow 0.15s ease;
    }

    #send-btn:hover {
      transform: translateY(-1px);
      box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4);
    }

    #send-btn:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }

    .cursor-blink {
      display: inline-block;
      width: 7px;
      height: 15px;
      background: #8b5cf6;
      margin-left: 3px;
      vertical-align: -2px;
      animation: blink 0.8s infinite;
    }

    @keyframes blink {
      0%, 50% { opacity: 1; }
      51%, 100% { opacity: 0; }
    }
  </style>
</head>
<body>
  <main class="chat-container">
    <header>
      <div class="brand-title">
        <div class="brand-logo">S1</div>
        <div class="brand-text">
          <h1>System One Engine</h1>
          <p>Dual-Process: 3ms Decision Reflex + System 2 Synthesis</p>
        </div>
      </div>
      <div class="header-badges">
        <div class="badge-status">System 1: INT8 Ready</div>
        <div class="badge-s2">System 2: Qwen2.5-Coder</div>
      </div>
    </header>

    <div id="messages-box">
      <div class="message-row bot">
        <div class="bubble bot">
          👋 Welcome to the <strong>Dual-Process System One & Two Engine</strong>! 
          <br><br>
          ⚡ <strong>System 1 (3 ms):</strong> Performs instant non-autoregressive decision triage (Intent, Task Complexity, and Safety Guardrails).
          <br>
          🧠 <strong>System 2 (Ollama Qwen2.5-Coder):</strong> Synthesizes real, runnable Python code, debugs errors, or writes full explanations steered by System 1.
        </div>
      </div>
    </div>

    <!-- Suggested Quick Prompts -->
    <div class="chips-tray">
      <button class="chip-btn" onclick="sendPrompt('give me code for a simple python snake game')">🎮 Snake Game (Python)</button>
      <button class="chip-btn" onclick="sendPrompt('write a python function to check if a number is prime')">🔢 Prime Checker</button>
      <button class="chip-btn" onclick="sendPrompt('My package arrived damaged with broken glass and I want a refund right away!')">📦 Refund Triage (Urgent)</button>
      <button class="chip-btn" onclick="sendPrompt('How do I safely quantize a PyTorch transformer model to INT8 ONNX?')">⚡ INT8 Export Question</button>
    </div>

    <footer>
      <input type="text" id="chat-input" placeholder="Ask for code, debugging, or general inquiries..." autocomplete="off" />
      <button id="send-btn" onclick="handleSend()">Send</button>
    </footer>
  </main>

  <script>
    const inputField = document.getElementById('chat-input');
    const messagesBox = document.getElementById('messages-box');
    const sendBtn = document.getElementById('send-btn');

    inputField.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') handleSend();
    });

    function sendPrompt(text) {
      inputField.value = text;
      handleSend();
    }

    function formatMarkdown(text) {
      if (!text) return '';

      // Format complete code blocks ```lang\ncode\n```
      let formatted = text.replace(/```([a-zA-Z0-9_-]*)\r?\n([\s\S]*?)```/g, (match, lang, code) => {
        const cleanLang = lang || 'code';
        const escaped = code
          .replace(/&/g, '&amp;')
          .replace(/</g, '&lt;')
          .replace(/>/g, '&gt;');
        return `###CB_START###<pre class="code-block"><div class="code-header"><span>${cleanLang}</span><button class="copy-btn" onclick="copyCode(this)">Copy</button></div><code>${escaped}</code></pre>###CB_END###`;
      });

      // Format unclosed code block if currently streaming
      const openCodeMatch = formatted.match(/```([a-zA-Z0-9_-]*)\r?\n([\s\S]*)$/);
      if (openCodeMatch) {
        const lang = openCodeMatch[1] || 'code';
        const code = openCodeMatch[2]
          .replace(/&/g, '&amp;')
          .replace(/</g, '&lt;')
          .replace(/>/g, '&gt;');
        const streamingBlock = `###CB_START###<pre class="code-block"><div class="code-header"><span>${lang}</span><button class="copy-btn" onclick="copyCode(this)">Copy</button></div><code>${code}</code></pre>###CB_END###`;
        formatted = formatted.substring(0, openCodeMatch.index) + streamingBlock;
      }

      // Format inline code `code`
      formatted = formatted.replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>');

      // Convert linebreaks outside codeblocks
      const parts = formatted.split(/(###CB_START[\s\S]*?###CB_END###)/);
      for (let i = 0; i < parts.length; i++) {
        if (parts[i].startsWith('###CB_START###')) {
          parts[i] = parts[i].replace('###CB_START###', '').replace('###CB_END###', '');
        } else {
          parts[i] = parts[i].replace(/\r?\n/g, '<br>');
        }
      }
      return parts.join('');
    }

    function copyCode(btn) {
      const code = btn.closest('.code-block').querySelector('code').innerText;
      navigator.clipboard.writeText(code).then(() => {
        btn.textContent = 'Copied!';
        setTimeout(() => { btn.textContent = 'Copy'; }, 2000);
      });
    }

    async function handleSend() {
      const text = inputField.value.trim();
      if (!text) return;

      appendMessage('user', text);
      inputField.value = '';
      sendBtn.disabled = true;

      // Create bot response container
      const botRow = document.createElement('div');
      botRow.className = 'message-row bot';
      const bubble = document.createElement('div');
      bubble.className = 'bubble bot';
      bubble.innerHTML = '<span class="cursor-blink"></span>';
      botRow.appendChild(bubble);
      messagesBox.appendChild(botRow);
      messagesBox.scrollTop = messagesBox.scrollHeight;

      let rawContent = '';
      let telemetryRendered = false;

      try {
        const response = await fetch('/chat/stream', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text })
        });

        if (!response.ok) {
          throw new Error('Server returned HTTP ' + response.status);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          // Split on SSE event boundary (double newline)
          const events = buffer.split(/\r?\n\r?\n/);
          buffer = events.pop() || ''; // retain trailing partial event

          for (const ev of events) {
            const lines = ev.split(/\r?\n/);
            for (const line of lines) {
              const trimmedLine = line.trim();
              if (!trimmedLine.startsWith('data:')) continue;
              const jsonStr = trimmedLine.substring(5).trim();
              if (!jsonStr) continue;

              try {
                const payload = JSON.parse(jsonStr);

                // Render System 1 telemetry card as soon as it arrives (Chunk 1)
                if (payload.telemetry && !telemetryRendered) {
                  telemetryRendered = true;
                  renderTelemetryCard(botRow, payload.telemetry);
                }

                if (payload.delta) {
                  rawContent += payload.delta;
                  bubble.innerHTML = formatMarkdown(rawContent) + '<span class="cursor-blink"></span>';
                  messagesBox.scrollTop = messagesBox.scrollHeight;
                }

                if (payload.done) {
                  bubble.innerHTML = formatMarkdown(rawContent);
                }
              } catch (parseErr) {
                console.warn('JSON parse skipped invalid fragment:', parseErr, jsonStr);
              }
            }
          }
        }

        bubble.innerHTML = formatMarkdown(rawContent);
      } catch (err) {
        bubble.innerHTML = '⚠️ Error: ' + err.message;
      } finally {
        sendBtn.disabled = false;
        inputField.focus();
        messagesBox.scrollTop = messagesBox.scrollHeight;
      }
    }

    function appendMessage(role, text) {
      const row = document.createElement('div');
      row.className = 'message-row ' + role;
      const bubble = document.createElement('div');
      bubble.className = 'bubble ' + role;
      bubble.textContent = text;
      row.appendChild(bubble);
      messagesBox.appendChild(row);
      messagesBox.scrollTop = messagesBox.scrollHeight;
    }

    function renderTelemetryCard(parentRow, telemetry) {
      const card = document.createElement('div');
      card.className = 'telemetry-card';

      const escClass = telemetry.needs_escalation ? 'pill-urgent' : 'pill-normal';
      const escText = telemetry.needs_escalation ? 'GUARD INTERCEPTION' : 'PASSED (SAFE)';

      card.innerHTML = `
        <div class="telemetry-header">
          <div class="telemetry-title">
            ⚡ System 1 Decision Reflex
          </div>
          <div class="telemetry-latency">
            ${telemetry.system_one_latency_ms.toFixed(2)} ms
          </div>
        </div>
        <div class="telemetry-grid">
          <div class="tag-item">
            <span class="tag-label">Classified Intent</span>
            <span class="tag-val" style="color: var(--accent-purple);">${telemetry.intent} (${(telemetry.intent_confidence * 100).toFixed(1)}%)</span>
          </div>
          <div class="tag-item">
            <span class="tag-label">Task Complexity</span>
            <span class="tag-val" style="color: var(--accent-amber);">${telemetry.sentiment_score.toFixed(2)} / 4.0 (${telemetry.sentiment_level})</span>
          </div>
          <div class="tag-item">
            <span class="tag-label">Safety Policy Check</span>
            <span class="tag-val ${escClass}">${escText}</span>
          </div>
        </div>
      `;
      parentRow.appendChild(card);
    }
  </script>
</body>
</html>
"""
