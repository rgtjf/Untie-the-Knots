(function () {
  'use strict';

  const LS_LANG = 'utk.lang';
  const LS_THEME = 'utk.theme';

  const body = document.body;

  // ---------- Language toggle ----------
  function applyLang(lang) {
    body.classList.remove('lang-en', 'lang-zh');
    body.classList.add('lang-' + lang);
    document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en';
    try { localStorage.setItem(LS_LANG, lang); } catch (e) { /* ignore */ }
  }

  const savedLang = (function () {
    try { return localStorage.getItem(LS_LANG); } catch (e) { return null; }
  })();
  applyLang(savedLang === 'zh' ? 'zh' : 'en');

  const langBtn = document.getElementById('lang-toggle');
  if (langBtn) {
    langBtn.addEventListener('click', function () {
      const current = body.classList.contains('lang-zh') ? 'zh' : 'en';
      applyLang(current === 'zh' ? 'en' : 'zh');
    });
  }

  // ---------- Theme toggle ----------
  function applyTheme(theme) {
    if (theme === 'dark') body.classList.add('dark');
    else body.classList.remove('dark');
    try { localStorage.setItem(LS_THEME, theme); } catch (e) { /* ignore */ }
  }

  const savedTheme = (function () {
    try { return localStorage.getItem(LS_THEME); } catch (e) { return null; }
  })();
  applyTheme(savedTheme === 'dark' ? 'dark' : 'light');

  const themeBtn = document.getElementById('theme-toggle');
  if (themeBtn) {
    themeBtn.addEventListener('click', function () {
      applyTheme(body.classList.contains('dark') ? 'light' : 'dark');
    });
  }

  // ---------- BibTeX copy ----------
  const copyBtn = document.getElementById('copy-bib');
  const bib = document.getElementById('bibtex');
  if (copyBtn && bib) {
    copyBtn.addEventListener('click', async function () {
      const text = bib.innerText;
      try {
        await navigator.clipboard.writeText(text);
      } catch (e) {
        const ta = document.createElement('textarea');
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        try { document.execCommand('copy'); } catch (_) { /* ignore */ }
        document.body.removeChild(ta);
      }
      copyBtn.classList.add('copied');
      const origHTML = copyBtn.innerHTML;
      copyBtn.innerHTML = '<span class="en">Copied!</span><span class="zh">已复制！</span>';
      setTimeout(function () {
        copyBtn.classList.remove('copied');
        copyBtn.innerHTML = origHTML;
      }, 1800);
    });
  }

  // ---------- IntersectionObserver: reveal on scroll ----------
  if ('IntersectionObserver' in window) {
    const io = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
          io.unobserve(entry.target);
        }
      });
    }, { threshold: 0.08 });
    document.querySelectorAll('.section, .kpi, .method-card, .analysis-card, .model-card').forEach(function (el) {
      el.classList.add('reveal');
      io.observe(el);
    });
  }
})();
