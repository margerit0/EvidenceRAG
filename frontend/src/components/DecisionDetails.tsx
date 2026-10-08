import type { Decision, Invocation } from '../lib/model';

const decisionNames: Record<Decision['action'], string> = {
  search_docs: '搜索文档',
  read_passage: '读取证据',
  answer: '提交答案',
  clarify: '请求补充信息',
  abstain: '停止作答',
};

export function DecisionDetails({ call, running }: { call: Invocation; running: boolean }) {
  if (call.action !== 'decide') return null;
  const decision = call.decision;
  return (
    <section className="decision-detail" aria-label="决策内容">
      <h3>决策内容</h3>
      {decision ? (
        <>
          <dl>
            <div className="decision-action">
              <dt>选择动作</dt>
              <dd>{decisionNames[decision.action]}</dd>
            </div>
            {decision.action === 'search_docs' && (
              <div>
                <dt>搜索词</dt>
                <dd>{decision.query}</dd>
              </div>
            )}
            {decision.action === 'read_passage' && (
              <div>
                <dt>证据编号</dt>
                <dd>段落 {decision.evidence_id}</dd>
              </div>
            )}
            {decision.action === 'clarify' && (
              <div>
                <dt>追问内容</dt>
                <dd>{decision.question}</dd>
              </div>
            )}
          </dl>
          {decision.action === 'answer' && <p>提交回答，进入答案与引用校验。</p>}
          {decision.action === 'abstain' && <p>结束本次调查，不生成答案。</p>}
        </>
      ) : (
        <p>
          {call.end === undefined
            ? running
              ? '正在生成决策，返回后会显示所选动作与参数。'
              : '调用未完成，尚无决策内容。'
            : call.outcome && call.outcome !== 'ok'
              ? '本次调用未产生有效决策，详情见返回状态。'
              : '此记录未提供决策详情。'}
        </p>
      )}
    </section>
  );
}
