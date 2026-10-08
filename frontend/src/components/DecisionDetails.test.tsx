import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { DecisionDetails } from './DecisionDetails';
import type { Invocation } from '../lib/model';

const pending: Invocation = {
  id: 'call-1',
  action: 'decide',
  node: 'agent',
  step: 1,
  start: 0,
  evidenceIds: [],
};

describe('decision availability', () => {
  it('distinguishes waiting, interrupted, failed, and legacy records', () => {
    expect(renderToStaticMarkup(<DecisionDetails call={pending} running />)).toContain(
      '正在生成决策',
    );
    expect(renderToStaticMarkup(<DecisionDetails call={pending} running={false} />)).toContain(
      '调用未完成',
    );
    expect(
      renderToStaticMarkup(
        <DecisionDetails
          call={{ ...pending, end: 1, outcome: 'invalid_action' }}
          running={false}
        />,
      ),
    ).toContain('未产生有效决策');
    expect(
      renderToStaticMarkup(
        <DecisionDetails call={{ ...pending, end: 1, outcome: 'ok' }} running={false} />,
      ),
    ).toContain('此记录未提供决策详情');
  });
});
