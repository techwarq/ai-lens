import { test } from 'node:test'
import * as assert from 'node:assert/strict'
import * as fs from 'fs'
import * as path from 'path'
import { resolveConfig } from '../src/config'
import { tempLogDir } from './helpers'

function withConfigFile(contents: object): string {
  const dir = tempLogDir()
  fs.writeFileSync(path.join(dir, 'config.json'), JSON.stringify(contents))
  return dir
}

test('priority: code > env > config.json > defaults', () => {
  const logDir = withConfigFile({ analysisModel: 'from-file', maxLogs: 5, verbose: true })

  const fileOnly = resolveConfig({ logDir }, {})
  assert.equal(fileOnly.analysisModel, 'from-file')
  assert.equal(fileOnly.maxLogs, 5)

  const envWins = resolveConfig({ logDir }, { AILENS_MODEL: 'from-env' })
  assert.equal(envWins.analysisModel, 'from-env')

  const codeWins = resolveConfig({ logDir, analysisModel: 'from-code' }, { AILENS_MODEL: 'from-env' })
  assert.equal(codeWins.analysisModel, 'from-code')

  const defaults = resolveConfig({ logDir: tempLogDir() }, {})
  assert.equal(defaults.analysisProvider, 'anthropic')
  assert.equal(defaults.checkErrors, 'fail')
  assert.equal(defaults.maxLogs, 1000)
})

test('undefined code values do not erase env/file values', () => {
  const cfg = resolveConfig({ logDir: tempLogDir(), analysisModel: undefined }, { AILENS_MODEL: 'env-model' })
  assert.equal(cfg.analysisModel, 'env-model')
})

test('API key is chosen for the selected provider', () => {
  const env = { ANTHROPIC_API_KEY: 'ant', OPENAI_API_KEY: 'oai' }
  const logDir = tempLogDir()
  assert.equal(resolveConfig({ logDir }, env).analysisApiKey, 'ant')
  assert.equal(resolveConfig({ logDir, analysisProvider: 'openai' }, env).analysisApiKey, 'oai')
  assert.equal(resolveConfig({ logDir }, { ...env, AILENS_PROVIDER: 'openai' }).analysisApiKey, 'oai')
  assert.equal(resolveConfig({ logDir }, { ...env, AILENS_API_KEY: 'generic' }).analysisApiKey, 'generic')
})

test('invalid AILENS_PROVIDER is ignored', () => {
  const cfg = resolveConfig({ logDir: tempLogDir() }, { AILENS_PROVIDER: 'gemini' })
  assert.equal(cfg.analysisProvider, 'anthropic')
})

test('config.json model is not reused after switching provider', () => {
  const logDir = withConfigFile({ analysisProvider: 'anthropic', analysisModel: 'claude-sonnet-4-6' })
  assert.equal(resolveConfig({ logDir }, {}).analysisModel, 'claude-sonnet-4-6')
  assert.equal(resolveConfig({ logDir }, { AILENS_PROVIDER: 'openai' }).analysisModel, undefined)
  assert.equal(resolveConfig({ logDir, analysisProvider: 'openai' }, {}).analysisModel, undefined)
})
