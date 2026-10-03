/** Poll until pollFn returns non-null or the timeout elapses. Returns null on timeout. */
export async function pollUntilDone(
  jobId: string,
  pollFn: (jobId: string) => Promise<string | null>,
  pollInterval: number,
  timeout: number,
  onPending?: (elapsedMs: number) => void
): Promise<string | null> {
  const start = Date.now()
  const deadline = start + timeout
  while (Date.now() < deadline) {
    await new Promise(r => setTimeout(r, pollInterval))
    const result = await pollFn(jobId)
    if (result !== null) return result
    onPending?.(Date.now() - start)
  }
  return null
}
