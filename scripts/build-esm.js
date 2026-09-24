// Generates dist/esm/index.js: a thin ES module wrapper around the CommonJS build.
// One implementation for both `import` and `require` means no dual-package hazard
// (e.g. `instanceof AILensCheckError` works no matter how the package was loaded).
const fs = require('fs')
const path = require('path')

const dist = path.join(__dirname, '..', 'dist')
const names = Object.keys(require(path.join(dist, 'index.js'))).filter(n => n !== '__esModule')

const out = path.join(dist, 'esm')
fs.rmSync(out, { recursive: true, force: true })
fs.mkdirSync(out, { recursive: true })
fs.writeFileSync(path.join(out, 'package.json'), '{"type":"module"}\n')
fs.writeFileSync(
  path.join(out, 'index.js'),
  `import cjs from '../index.js'\n\nexport const {\n${names.map(n => `  ${n},`).join('\n')}\n} = cjs\n`
)
console.log(`dist/esm/index.js: ${names.join(', ')}`)
