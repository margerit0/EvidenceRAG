import { expect, test } from '@playwright/test';
import { resumableDemo } from '../src/lib/demo';

for (const [status, reason, message] of [
  ['unrecoverable', 'call_outcome_unknown', '无法确认外部调用是否完成'],
  ['unrecoverable', 'snapshot_or_model_mismatch', '原模型配置或文档快照已不可用'],
  ['budget_exhausted', 'budget_exhausted', '累计执行预算已耗尽'],
]) {
  test(`persistent terminal reason is actionable: ${reason}`, async ({ page }) => {
    const run = {
      ...resumableDemo('synthetic-terminal', '合成恢复失败场景').durable!,
      status,
      reason,
    };
    await page.addInitScript(() => {
      localStorage.setItem('zhrag-run-mode', 'live');
      localStorage.setItem('zhrag-theme', 'dark');
    });
    await page.route('**/api/capabilities', (route) =>
      route.fulfill({
        json: {
          agent_enabled: true,
          agent_profile: 'synthetic',
          agent_persistence: true,
          agent_models: [{ id: run.model, name: run.model }],
          default_agent_model: run.model,
        },
      }),
    );
    await page.route('**/api/investigations', (route) =>
      route.fulfill({
        json: {
          runs: [
            {
              id: run.run_id,
              query: run.query,
              model: run.model,
              status,
              reason,
              version: run.version,
              created_at: run.created_at,
            },
          ],
        },
      }),
    );
    await page.route(`**/api/investigations/${run.run_id}`, (route) =>
      route.fulfill({ json: run }),
    );
    await page.goto('/workbench/');
    await expect(page.getByText(message, { exact: false })).toBeVisible();
    await expect(page.getByRole('button', { name: '继续调查', exact: true })).toHaveCount(0);
    await expect(page.getByRole('button', { name: '提交补充并继续' })).toHaveCount(0);
  });
}
