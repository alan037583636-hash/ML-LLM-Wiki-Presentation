import sys
import json
sys.path.append('C:/Users/User/Desktop/llm-wiki-project/scratch')
from verify_slides import slides_data
from generate_svgs import svg_dict

# Build modalData JSON safely
modal_dict = {}
for s in slides_data:
    modal_dict[str(s['id'])] = {
        "title": s['modal_title'],
        "content": s['modal_content']
    }

modal_json_str = json.dumps(modal_dict, ensure_ascii=False)

html_template = """<!DOCTYPE html>
<html lang="zh-TW">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>機器學習基礎與模型演進 · 教學簡報</title>
  <!-- KaTeX CSS & JS -->
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"></script>
  <!-- Tailwind CSS CDN -->
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = {
      theme: {
        extend: {
          colors: {
            brand: {
              dark: '#0b0f19',
              card: '#111827',
              border: '#1f293d',
              cyan: '#00f2fe',
              blue: '#38bdf8',
              accent: '#3b82f6'
            }
          }
        }
      }
    }
  </script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background-color: #070a12;
      color: #f1f5f9;
      overflow: hidden;
      user-select: none;
    }
    .slide-container {
      width: 100vw;
      height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      position: relative;
    }
    .slide-card {
      width: 95vw;
      max-width: 1280px;
      height: 88vh;
      max-height: 720px;
      background: radial-gradient(circle at 50% 0%, #151d30 0%, #0d121f 100%);
      border: 1px solid #1e293b;
      box-shadow: 0 25px 50px -12px rgba(0, 242, 254, 0.08), 0 0 30px rgba(0, 0, 0, 0.8);
      border-radius: 16px;
      display: none;
      flex-direction: column;
      position: relative;
      overflow: hidden;
    }
    .slide-card.active {
      display: flex;
    }
    .code-font {
      font-family: 'JetBrains Mono', monospace;
    }
    /* Modal styles */
    .modal-backdrop {
      background: rgba(3, 7, 18, 0.85);
      backdrop-filter: blur(8px);
    }
    /* Custom scrollbar for modal */
    .modal-scroll::-webkit-scrollbar {
      width: 6px;
    }
    .modal-scroll::-webkit-scrollbar-track {
      background: #111827;
    }
    .modal-scroll::-webkit-scrollbar-thumb {
      background: #374151;
      border-radius: 3px;
    }
  </style>
</head>
<body class="w-screen h-screen">

  <div class="slide-container">
    <!-- Top Progress Bar -->
    <div class="absolute top-0 left-0 w-full h-1.5 bg-gray-900 z-50">
      <div id="progress-bar" class="h-full bg-gradient-to-r from-cyan-400 to-blue-500 transition-all duration-300" style="width: 5%;"></div>
    </div>

    <!-- Navigation Header Controls -->
    <header class="absolute top-4 left-8 right-8 flex justify-between items-center text-xs text-gray-400 z-40">
      <div class="flex items-center space-x-3">
        <span class="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-pulse"></span>
        <span class="font-semibold tracking-wider text-gray-300 uppercase">ML Foundations & Model Evolution</span>
      </div>
      <div class="flex items-center space-x-4">
        <span id="slide-indicator" class="code-font bg-gray-800/80 px-3 py-1 rounded-full border border-gray-700 text-cyan-300">
          Slide 01 / 19
        </span>
        <button onclick="toggleHelp()" class="hover:text-cyan-400 transition" title="操作說明">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
        </button>
      </div>
    </header>

    <!-- Slide Cards Container -->
    <main id="slides-wrapper" class="relative z-10 w-full flex justify-center">
"""

slides_html = []
for s in slides_data:
    sid = s['id']
    badge = s['badge']
    title = s['title']
    subtitle = s['subtitle']
    points = s['points']
    svg = svg_dict.get(sid, '')
    
    points_li = "".join([f'<li class="flex items-start space-x-3 text-sm text-gray-300 leading-relaxed"><span class="w-1.5 h-1.5 rounded-full bg-cyan-400 mt-2 flex-shrink-0"></span><span>{p}</span></li>' for p in points])

    card = f"""
      <!-- Slide {sid:02d} -->
      <section class="slide-card {'active' if sid==1 else ''}" id="slide-{sid}" data-slide="{sid}">
        <!-- Slide Header -->
        <div class="px-8 pt-6 pb-3 border-b border-gray-800 flex justify-between items-baseline">
          <div>
            <span class="inline-block px-2.5 py-0.5 rounded text-[11px] font-semibold bg-cyan-950/80 text-cyan-400 border border-cyan-800/60 mb-1">
              {badge}
            </span>
            <h1 class="text-xl md:text-2xl font-bold text-white tracking-wide">{title}</h1>
            <p class="text-xs text-gray-400 tracking-wider font-medium">{subtitle}</p>
          </div>
          <span class="code-font text-xs text-gray-500">#{sid:02d}</span>
        </div>

        <!-- Slide Body (5:5 Grid) -->
        <div class="flex-1 grid grid-cols-1 md:grid-cols-2 gap-6 p-6 items-center overflow-hidden">
          <!-- Left: SVG Diagram Container -->
          <div class="w-full h-full max-h-[360px] bg-gray-950/60 rounded-xl border border-gray-800/80 flex items-center justify-center p-3 relative group overflow-hidden">
            {svg}
          </div>

          <!-- Right: Key Points Cards -->
          <div class="flex flex-col justify-between h-full max-h-[360px] py-1">
            <div class="bg-gray-900/50 rounded-xl p-5 border border-gray-800/60">
              <h3 class="text-xs font-semibold text-cyan-300 tracking-wider uppercase mb-3 flex items-center space-x-2">
                <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                <span>核心概念直覺 (Core Intuition)</span>
              </h3>
              <ul class="space-y-3">
                {points_li}
              </ul>
            </div>

            <!-- Bottom Action Row -->
            <div class="flex justify-between items-center pt-2">
              <span class="text-[11px] text-gray-500 italic">按 [←] [→] 鍵切換投影片</span>
              <button onclick="openModal({sid})" class="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-400 border border-cyan-500/30 text-xs font-medium transition duration-200">
                <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"></path></svg>
                <span>深入數學推導</span>
              </button>
            </div>
          </div>
        </div>

        <!-- Slide Footer Indicator -->
        <div class="px-8 py-2.5 bg-gray-950/80 border-t border-gray-900 flex justify-between items-center text-[11px] text-gray-500">
          <span>Machine Learning Foundations · Chapter 1~7 Ingest</span>
          <span class="code-font">{sid} / 19</span>
        </div>
      </section>
    """
    slides_html.append(card)

html_end = f"""
    </main>

    <!-- Bottom Navigation Controls -->
    <footer class="absolute bottom-4 left-8 right-8 flex justify-between items-center z-40">
      <div class="flex space-x-2">
        <button onclick="prevSlide()" class="p-2 rounded-lg bg-gray-900/80 border border-gray-800 text-gray-300 hover:text-cyan-400 hover:border-cyan-500/50 transition">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"></path></svg>
        </button>
        <button onclick="nextSlide()" class="p-2 rounded-lg bg-gray-900/80 border border-gray-800 text-gray-300 hover:text-cyan-400 hover:border-cyan-500/50 transition">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path></svg>
        </button>
      </div>

      <div class="text-[11px] text-gray-400 code-font">
        Use Keyboard: <kbd class="px-1.5 py-0.5 rounded bg-gray-800 border border-gray-700">←</kbd> <kbd class="px-1.5 py-0.5 rounded bg-gray-800 border border-gray-700">→</kbd> or <kbd class="px-1.5 py-0.5 rounded bg-gray-800 border border-gray-700">Space</kbd>
      </div>
    </footer>
  </div>

  <!-- Math Modal Component -->
  <div id="math-modal" class="fixed inset-0 z-50 modal-backdrop hidden items-center justify-center p-4">
    <div class="bg-gray-900 border border-cyan-500/30 rounded-2xl max-w-2xl w-full max-h-[85vh] flex flex-col shadow-2xl shadow-cyan-950/50 overflow-hidden">
      <!-- Modal Header -->
      <div class="px-6 py-4 border-b border-gray-800 flex justify-between items-center bg-gray-950/60">
        <div class="flex items-center space-x-2 text-cyan-400">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z"></path></svg>
          <h3 id="modal-title" class="font-bold text-sm tracking-wide text-white">數學推導與深度解析</h3>
        </div>
        <button onclick="closeModal()" class="text-gray-400 hover:text-white transition">
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path></svg>
        </button>
      </div>
      <!-- Modal Body -->
      <div id="modal-body" class="p-6 overflow-y-auto modal-scroll text-sm text-gray-300 leading-relaxed space-y-4">
        <!-- Injected dynamically -->
      </div>
      <!-- Modal Footer -->
      <div class="px-6 py-3 border-t border-gray-800 bg-gray-950/40 flex justify-between items-center text-xs text-gray-500">
        <span>按 <kbd class="px-1.5 py-0.5 rounded bg-gray-800 border border-gray-700">ESC</kbd> 鍵關閉視窗</span>
        <button onclick="closeModal()" class="px-4 py-1.5 rounded-lg bg-gray-800 hover:bg-gray-700 text-gray-200 text-xs transition">
          關閉 (Close)
        </button>
      </div>
    </div>
  </div>

  <!-- Help Modal -->
  <div id="help-modal" class="fixed inset-0 z-50 modal-backdrop hidden items-center justify-center p-4">
    <div class="bg-gray-900 border border-gray-700 rounded-xl max-w-md w-full p-6 text-sm text-gray-300">
      <h3 class="text-base font-bold text-white mb-3 flex items-center space-x-2">
        <span class="w-2 h-2 rounded-full bg-cyan-400"></span>
        <span>簡報快捷鍵說明</span>
      </h3>
      <ul class="space-y-2 mb-6 text-xs text-gray-400">
        <li class="flex justify-between"><span class="text-gray-200">[→] 或 [Space]</span> <span>下一頁 (Next Slide)</span></li>
        <li class="flex justify-between"><span class="text-gray-200">[←]</span> <span>上一頁 (Previous Slide)</span></li>
        <li class="flex justify-between"><span class="text-gray-200">[M]</span> <span>開啟當前頁數學推導 Modal</span></li>
        <li class="flex justify-between"><span class="text-gray-200">[ESC]</span> <span>關閉彈出視窗 (Close Modal)</span></li>
      </ul>
      <button onclick="toggleHelp()" class="w-full py-2 bg-cyan-600 hover:bg-cyan-500 text-white font-medium rounded-lg text-xs transition">我知道了</button>
    </div>
  </div>

  <!-- Embedded JSON Data Script -->
  <script id="modal-data-json" type="application/json">
{modal_json_str}
  </script>

  <!-- Navigation and Modal Script -->
  <script>
    let modalData = {{}};
    try {{
      const jsonText = document.getElementById('modal-data-json').textContent;
      modalData = JSON.parse(jsonText);
    }} catch (e) {{
      console.error('Failed to parse modal data JSON:', e);
    }}

    let currentSlide = 1;
    const totalSlides = 19;

    function renderMath(targetElement) {{
      if (window.renderMathInElement && targetElement) {{
        renderMathInElement(targetElement, {{
          delimiters: [
            {{left: '$$', right: '$$', display: true}},
            {{left: '$', right: '$', display: false}}
          ],
          throwOnError: false
        }});
      }}
    }}

    function updateSlide() {{
      document.querySelectorAll('.slide-card').forEach(el => el.classList.remove('active'));
      const activeCard = document.getElementById('slide-' + currentSlide);
      if (activeCard) {{
        activeCard.classList.add('active');
        renderMath(activeCard);
      }}
      // Update progress
      const percent = (currentSlide / totalSlides) * 100;
      document.getElementById('progress-bar').style.width = percent + '%';
      document.getElementById('slide-indicator').textContent = 'Slide ' + String(currentSlide).padStart(2, '0') + ' / ' + totalSlides;
    }}

    function nextSlide() {{
      if (currentSlide < totalSlides) {{
        currentSlide++;
        updateSlide();
      }}
    }}

    function prevSlide() {{
      if (currentSlide > 1) {{
        currentSlide--;
        updateSlide();
      }}
    }}

    function openModal(id) {{
      const data = modalData[String(id)];
      if (!data) return;
      document.getElementById('modal-title').textContent = data.title;
      const body = document.getElementById('modal-body');
      body.innerHTML = data.content;
      
      const modal = document.getElementById('math-modal');
      modal.classList.remove('hidden');
      modal.classList.add('flex');

      renderMath(body);
    }}

    function closeModal() {{
      const modal = document.getElementById('math-modal');
      modal.classList.add('hidden');
      modal.classList.remove('flex');
    }}

    function toggleHelp() {{
      const h = document.getElementById('help-modal');
      h.classList.toggle('hidden');
      h.classList.toggle('flex');
    }}

    // Keyboard navigation
    window.addEventListener('keydown', (e) => {{
      const modalOpen = !document.getElementById('math-modal').classList.contains('hidden');
      if (e.key === 'Escape') {{
        closeModal();
        document.getElementById('help-modal').classList.add('hidden');
        document.getElementById('help-modal').classList.remove('flex');
      }} else if (!modalOpen) {{
        if (e.key === 'ArrowRight' || e.key === ' ') {{
          e.preventDefault();
          nextSlide();
        }} else if (e.key === 'ArrowLeft') {{
          e.preventDefault();
          prevSlide();
        }} else if (e.key === 'm' || e.key === 'M') {{
          openModal(currentSlide);
        }}
      }}
    }});

    // Initial render
    window.addEventListener('DOMContentLoaded', () => {{
      updateSlide();
      // Ensure Math renders when KaTeX loads
      setTimeout(() => {{
        const activeCard = document.getElementById('slide-' + currentSlide);
        renderMath(activeCard);
      }}, 500);
    }});
  </script>
</body>
</html>
"""

full_html = html_template + "".join(slides_html) + html_end
with open("C:/Users/User/Desktop/llm-wiki-project/index.html", "w", encoding="utf-8") as f:
    f.write(full_html)
print("Successfully wrote repaired index.html. Size:", len(full_html))
