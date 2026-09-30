import { describe, expect, it } from 'vitest';
import { batchProgress, invocations, safeSourceUrl } from './model';
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
