// ============================================================
// 红源π · 液态玻璃主题 JS —— 光标即光源
// 布局不动，仅在玻璃元素上叠加跟随光标的高光
// ============================================================
(function () {
  function applyLight() {
    document.querySelectorAll('.relic-card, .glass-card-sub, .glass-card, .glass-button, .glass-button-primary').forEach(function (el) {
      el.addEventListener('mousemove', function (e) {
        var r = el.getBoundingClientRect();
        el.style.setProperty('--lx', (e.clientX - r.left) + 'px');
        el.style.setProperty('--ly', (e.clientY - r.top) + 'px');
      });
    });
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', applyLight);
  } else {
    applyLight();
  }
  // Vue 挂载后内容可能重渲染，延迟再绑一次
  setTimeout(applyLight, 600);
  setTimeout(applyLight, 2000);
})();
