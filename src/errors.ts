import { LensCall, CheckResult } from './types'

export class AILensCheckError extends Error {
  call: LensCall
  checks: CheckResult[]

  constructor(message: string, call: LensCall, checks: CheckResult[]) {
    super(message)
    this.name = 'AILensCheckError'
    this.call = call
    this.checks = checks
  }
}
