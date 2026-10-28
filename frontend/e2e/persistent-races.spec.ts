import { expect, test, type Page } from '@playwright/test';
import { resumableDemo } from '../src/lib/demo';
import type { DurableRun } from '../src/lib/model';

async function connect(page: Page, run: DurableRun) {
  await page.addInitScript(() => localStorage.setItem('zhrag-run-mode', 'live'));
  await page.route('**/api/capabilities', (route) =>
    route.fulfill({
      json: {
        agent_enabled: true,
        agent_persistence: true,
        agent_profile: 'synthetic',
        agent_models: [{ id: run.model, name: run.model }],
        default_agent_model: run.model,
      },
    }),
  );
  await page.route(`**/api/investigations/${run.run_id}`, (route) => route.fulfill({ json: run }));
  await page.route(`**/api/investigations/${run.run_id}/stream?**`, (route) =>
    route.fulfill({ contentType: 'text/event-stream', body: ': keep-alive\n\n' }),
  );
}

for (const status of ['paused', 'waiting_input']) {
  test(`demo mode keeps real ${status} history read-only`, async ({ page }) => {
    const run = { ...resumableDemo('real-history', '原创只读调查').durable!, status };
    await connect(page, run);
    let commands = 0;
    await page.route('**/api/investigations', (route) =>
      route.fulfill({ json: { runs: [{ ...run, id: run.run_id }] } }),
    );
    await page.route(`**/api/investigations/${run.run_id}/*`, (route) => {
      if (route.request().method() === 'POST') commands++;
      return route.fulfill({ json: run });
    });
    await page.goto('/workbench/');
    const button = page.getByRole('button', {
      name: status === 'paused' ? '继续调查' : '提交补充并继续',
      exact: true,
    });
    await expect(button).toBeVisible();
    await page.getByRole('combobox', { name: '运行模式' }).click();
    await page.getByRole('option', { name: '模拟演示', exact: true }).click();
    await expect(page.getByText('当前为真实调查的只读历史。', { exact: false })).toBeVisible();
    await expect(button).toBeDisabled();
    await expect(page.getByRole('button', { name: '取消调查', exact: true })).toBeDisabled();
    if (status === 'waiting_input')
      await expect(page.locator('.investigation-controls textarea')).toBeDisabled();
    await expect(page.getByRole('combobox', { name: '运行模式' })).toBeEnabled();
    expect(commands).toBe(0);
    await page.getByRole('combobox', { name: '运行模式' }).click();
    await page.getByRole('option', { name: '连接服务', exact: true }).click();
    await expect(page.getByRole('button', { name: '取消调查', exact: true })).toBeEnabled();
  });
}

test('late history cannot remove a newly created run or its subscription', async ({ page }) => {
  const run = {
    ...resumableDemo('created-during-list', '原创竞态问题').durable!,
    status: 'running',
    wait_id: null,
  };
  await connect(page, run);
  let release!: () => void;
  const held = new Promise<void>((resolve) => {
    release = resolve;
  });
  let listing = false;
  let reads = 0;
  await page.route('**/api/investigations', async (route) => {
    if (route.request().method() === 'POST') return route.fulfill({ json: run });
    listing = true;
    await held;
    return route.fulfill({ json: { runs: [] } });
  });
  await page.route(`**/api/investigations/${run.run_id}`, (route) => {
    reads++;
    return route.fulfill({ json: run });
  });
  await page.goto('/workbench/');
  await expect.poll(() => listing).toBe(true);
  await page.getByRole('textbox', { name: '调查问题' }).fill(run.query);
  await page.getByRole('button', { name: '开始调查', exact: true }).click();
  await expect(page.locator('.submitted-query')).toHaveText(run.query);
  const listed = page.waitForResponse(
    (response) =>
      response.url().endsWith('/api/investigations') && response.request().method() === 'GET',
  );
  release();
  await listed;
  await expect(page.getByRole('button', { name: '暂停调查', exact: true })).toBeVisible();
  const before = reads;
  await expect.poll(() => reads).toBeGreaterThan(before);
  await expect(page.locator('.submitted-query')).toHaveText(run.query);
});

for (const status of ['queued', 'resuming']) {
  test(`immediate pause survives ${status} claim before SSE`, async ({ page }) => {
    const run = { ...resumableDemo('pause-race', '原创暂停问题').durable!, status, wait_id: null };
    await connect(page, run);
    await page.route('**/api/investigations', (route) =>
      route.fulfill({ json: { runs: [{ ...run, id: run.run_id }] } }),
    );
    let posts = 0;
    const claimed = { ...run, status: 'running', version: run.version + 1, cursor: run.cursor + 1 };
    await page.route(`**/api/investigations/${run.run_id}`, (route) =>
      route.fulfill({ json: posts ? claimed : run }),
    );
    await page.route(`**/api/investigations/${run.run_id}/pause`, (route) => {
      posts++;
      if (posts === 1) {
        expect(route.request().postDataJSON().version).toBe(run.version);
        return route.fulfill({ status: 409, json: { code: 'stale_run_version' } });
      }
      expect(route.request().postDataJSON().version).toBe(claimed.version);
      return route.fulfill({
        json: {
          ...claimed,
          status: 'paused',
          version: claimed.version + 1,
          cursor: claimed.cursor + 1,
        },
      });
    });
    await page.goto('/workbench/');
    await page.getByRole('button', { name: '暂停调查', exact: true }).click();
    await expect(page.locator('.answer-status')).toContainText('已暂停');
    expect(posts).toBe(2);
  });
}

test('a pending real command keeps the mode and subscription attached', async ({ page }) => {
  const run = {
    ...resumableDemo('pending-command', '原创等待响应问题').durable!,
    status: 'paused',
  };
  await connect(page, run);
  await page.route('**/api/investigations', (route) =>
    route.fulfill({ json: { runs: [{ ...run, id: run.run_id }] } }),
  );
  let release!: () => void;
  const held = new Promise<void>((resolve) => {
    release = resolve;
  });
  await page.route(`**/api/investigations/${run.run_id}/resume`, async (route) => {
    await held;
    return route.fulfill({
      json: { ...run, status: 'resuming', version: run.version + 1, cursor: run.cursor + 1 },
    });
  });
  await page.goto('/workbench/');
  await page.getByRole('button', { name: '继续调查', exact: true }).click();
  await expect(page.getByRole('combobox', { name: '运行模式' })).toBeDisabled();
  release();
  await expect(page.locator('.answer-status')).toContainText('正在恢复');
  await expect(page.getByRole('combobox', { name: '运行模式' })).toBeDisabled();
});
