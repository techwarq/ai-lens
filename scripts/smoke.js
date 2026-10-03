// Verifies the built package loads via both require() and import, and the CLI starts.
const assert = require('assert')
const path = require('path')
const { execFileSync } = require('child_process')
const { pathToFileURL } = require('url')

const dist = path.join(__dirname, '..', 'dist')

;(async () => {
  const cjs = require(path.join(dist, 'index.js'))
  const esm = await import(pathToFileURL(path.join(dist, 'esm', 'index.js')).href)
  for (const name of ['lens', 'createLens', 'getDefaultLens', 'AILens', 'AILensCheckError']) {
    assert.equal(typeof cjs[name], 'function', `require: missing ${name}`)
    assert.equal(esm[name], cjs[name], `import: ${name} differs from require`)
  }
  const version = execFileSync(process.execPath, [path.join(dist, 'cli', 'index.js'), '--version']).toString().trim()
  assert.equal(version, require('../package.json').version)
  console.log('smoke ok')
})().catch(e => { console.error(e); process.exit(1) })
