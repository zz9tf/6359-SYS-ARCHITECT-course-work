// Renders the Lucide icons used on the slides to white 256px PNGs.
//   cd slides/assets && npm install react-icons react react-dom sharp && node render_icons.js
const React = require('react');
const { renderToStaticMarkup } = require('react-dom/server');
const sharp = require('sharp');
const lu = require('react-icons/lu');
const fs = require('fs');

const names = ['LuMemoryStick', 'LuShrink', 'LuGauge', 'LuRefreshCw', 'LuCpu', 'LuScale', 'LuBrain', 'LuBinary',
  'LuDatabase', 'LuSlidersHorizontal', 'LuActivity', 'LuServer', 'LuMessagesSquare', 'LuLightbulb', 'LuTarget',
  'LuBookOpen', 'LuTimer', 'LuZap', 'LuLayers', 'LuChartColumn', 'LuTrendingUp', 'LuTriangleAlert', 'LuCircleCheck',
  'LuFlaskConical', 'LuUsers', 'LuMicrochip', 'LuBattery', 'LuGitBranch'];
fs.mkdirSync('icons', { recursive: true });
(async () => {
  for (const n of names) {
    if (!lu[n]) { console.log('missing', n); continue; }
    const svg = renderToStaticMarkup(React.createElement(lu[n], { size: 256, color: '#FFFFFF', strokeWidth: 2 }));
    await sharp(Buffer.from(svg)).resize(256, 256).png().toFile(`icons/${n}.png`);
  }
  console.log('rendered', names.length);
})();
