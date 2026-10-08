import { describe, expect, it } from 'vitest';
import { batchProgress, invocations, progressSchema, resultSchema, safeSourceUrl } from './model';
import { demoScript, scenarios } from './demo';

describe('execution records', () => {
  it('keeps repeated tool invocations separate and pairs their start/end events', () => {
    const calls = invocations(demoScript('answered', 'run').events);
    const searches = calls.filter((call) => call.action === 'search_docs');
    expect(searches).toHaveLength(2);
    expect(searches[0].id).not.toBe(searches[1].id);
    expect(searches[0].evidenceIds).toEqual([1]);
    expect(searches[1].evidenceIds).toEqual([2]);
    expect(searches.every((call) => call.start !== undefined && call.end! > call.start)).toBe(true);
  });
  it('does not fabricate operation durations from legacy completion records', () => {
    const calls = invocations(batchProgress(demoScript('answered', 'run').result, 'legacy'));
    expect(calls.every((call) => call.start === undefined && call.end !== undefined)).toBe(true);
  });
  it.each(['stream', 'batch'])('preserves each decision through %s decoding', (transport) => {
    const script = demoScript('answered', 'run');
    const events =
      transport === 'stream'
        ? script.events.map((event) => progressSchema.parse(event))
        : batchProgress(resultSchema.parse(script.result), 'run');
    const decisions = invocations(events)
      .filter((call) => call.action === 'decide')
      .map((call) => call.decision);
    expect(decisions).toEqual([
      { action: 'search_docs', query: '示例集群 升级前 配置 兼容性' },
      { action: 'read_passage', evidence_id: 1 },
      { action: 'search_docs', query: '示例集群 升级演练 回退条件' },
      { action: 'read_passage', evidence_id: 2 },
      { action: 'answer' },
    ]);
  });
  it('accepts old records without inferring decision parameters from adjacent tools', () => {
    const { result } = demoScript('answered', 'run');
    for (const event of result.events) delete event.decision;
    const calls = invocations(batchProgress(resultSchema.parse(result), 'legacy'));
    expect(calls.filter((call) => call.action === 'decide').every((call) => !call.decision)).toBe(
      true,
    );
    expect(calls.some((call) => call.action === 'search_docs' && call.evidenceIds.length)).toBe(
      true,
    );
  });
  it('reveals decision content only after the matching completion event', () => {
    const { events } = demoScript('answered', 'run');
    expect(invocations(events.slice(0, 1))[0].decision).toBeUndefined();
    expect(invocations(events.slice(0, 2))[0].decision).toEqual({
      action: 'search_docs',
      query: '示例集群 升级前 配置 兼容性',
    });
    const parsed = progressSchema.parse({
      ...events[1],
      decision: { action: 'read_passage', evidence_id: 2, query: null, question: null },
    });
    expect(invocations([parsed])[0].decision).toEqual({ action: 'read_passage', evidence_id: 2 });
  });
  it.each(scenarios)(
    'keeps $id terminal status and publishes no incomplete answers',
    (scenario) => {
      const { events, result } = demoScript(scenario.id, 'run');
      expect(events.at(-1)?.outcome).toBe(scenario.id);
      expect(result.status).toBe(scenario.id);
      expect(!!result.blocks.length).toBe(scenario.id === 'answered');
      for (const block of result.blocks)
        for (const id of block.citations)
          expect(result.sources.some((source) => source.citation_id === id)).toBe(true);
    },
  );
  it('only permits safe source links', () => {
    expect(safeSourceUrl('javascript:alert(1)')).toBeUndefined();
    expect(safeSourceUrl('https://user:password@example.com/')).toBeUndefined();
    expect(safeSourceUrl(null)).toBeUndefined();
    expect(safeSourceUrl('https://example.com/docs')).toBe('https://example.com/docs');
  });
});
