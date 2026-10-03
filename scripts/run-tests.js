// Cross-platform test runner: finds test/*.test.ts and runs them with node:test via tsx.
const { spawnSync } = require('child_process')
const fs = require('fs')
const path = require('path')

const root = path.join(__dirname, '..')
const dir = path.join(root, 'test')
const files = fs.readdirSync(dir).filter(f => f.endsWith('.test.ts')).map(f => path.join('test', f))
const tsx = path.join(root, 'node_modules', 'tsx', 'dist', 'cli.mjs')

const res = spawnSync(process.execPath, [tsx, '--test', ...files], { stdio: 'inherit', cwd: root })
process.exit(res.status ?? 1)
