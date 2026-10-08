import { expect, test, type Page } from '@playwright/test';
import { demoScript, type ScenarioId } from '../src/lib/demo';

async function connect(
  page: Page,
  stream: boolean,
  scenario: ScenarioId = 'answered',
  legacy = false,
  searchQuery?: string,
) {
  const script = demoScript(scenario, 'decision-service');
  if (searchQuery) {
    for (const event of [...script.events, ...script.result.events]) {
      if (event.decision?.action === 'search_docs') event.decision.query = searchQuery;
    }
  }
  if (legacy) {
    for (const event of script.events) delete event.decision;
    for (const event of script.result.events) delete event.decision;
  }
  await page.route('**/api/**', (route) => route.abort());
  await page.route('**/api/capabilities', (route) =>
    route.fulfill({
      json: { agent_enabled: true, agent_profile: 'synthetic', agent_streaming: stream },
    }),
  );
  await page.route(stream ? '**/api/investigate/stream' : '**/api/investigate', (route) =>
    stream
      ? route.fulfill({
          contentType: 'text/event-stream',
          body:
            script.events
              .map((event) => `event: progress\ndata: ${JSON.stringify(event)}\n\n`)
              .join('') +
            `event: result\ndata: ${JSON.stringify({ run_id: 'decision-service', response: script.result })}\n\n`,
        })
      : route.fulfill({ json: script.result }),
  );
  await page.goto('/workbench/');
  await page.getByRole('combobox', { name: '运行模式' }).click();
  await page.getByRole('option', { name: '连接服务', exact: true }).click();
  await expect(page.locator('.connection-note')).toContainText('服务已连接');
  await page.getByRole('textbox', { name: '调查问题' }).fill('请核对合成系统的升级条件。');
  await page.getByRole('button', { name: '开始调查', exact: true }).click();
  return script;
}

for (const profile of [
  { name: 'stream desktop light', stream: true, width: 1440, theme: 'light' },
  { name: 'stream mobile dark', stream: true, width: 390, theme: 'dark' },
  { name: 'batch desktop dark', stream: false, width: 1440, theme: 'dark' },
  { name: 'batch mobile light', stream: false, width: 390, theme: 'light' },
]) {
  test(`shows the selected decision and its own parameters: ${profile.name}`, async ({
    page,
  }, info) => {
    await page.setViewportSize({
      width: profile.width,
      height: profile.width === 390 ? 844 : 1000,
    });
    await page.addInitScript((theme) => localStorage.setItem('zhrag-theme', theme), profile.theme);
    await connect(page, profile.stream);
    await expect(page.locator('.answer-status')).toContainText('已生成答案');
    await page.locator('.timeline-row').filter({ hasText: 'Agent 决策' }).first().click();
    const details = page.getByRole('region', { name: '决策内容' });
    await expect(details).toContainText('搜索文档');
    await expect(details).toContainText('示例集群 升级前 配置 兼容性');
    await expect(details).toBeInViewport({ ratio: 1 });
    await page.screenshot({ path: info.outputPath('decision-search.png') });
    const next = page.getByRole('button', { name: '下一次同类调用' });
    await next.click();
    await expect(details).toContainText('读取证据');
    await expect(details).toContainText('段落 1');
    await expect(details).not.toContainText('兼容性');
    await next.click();
    await expect(details).toContainText('示例集群 升级演练 回退条件');
    await next.click();
    await expect(details).toContainText('段落 2');
    await next.click();
    await expect(details).toContainText('提交答案');
    await expect(next).toBeDisabled();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(
      true,
    );
  });
}

for (const scenario of [
  'clarification_needed',
  'insufficient_evidence',
  'generation_timeout',
] as const) {
  test(`decision details for ${scenario}`, async ({ page }) => {
    const script = await connect(page, true, scenario);
    await expect(page.locator('.answer-status')).not.toContainText('调查进行中');
    await page.locator('.timeline-row').filter({ hasText: 'Agent 决策' }).last().click();
    const details = page.getByRole('region', { name: '决策内容' });
    if (scenario === 'clarification_needed') {
      await expect(details).toContainText('请求补充信息');
      await expect(details).toContainText(script.result.clarification);
    } else if (scenario === 'insufficient_evidence') {
      await expect(details).toContainText('停止作答');
    } else {
      await expect(details).toContainText('本次调用未产生有效决策');
    }
  });
}

test('older services display an explicit missing-details state', async ({ page }) => {
  await connect(page, false, 'answered', true);
  await expect(page.locator('.answer-status')).toContainText('已生成答案');
  await page.locator('.timeline-row').filter({ hasText: 'Agent 决策' }).first().click();
  await expect(page.getByRole('region', { name: '决策内容' })).toContainText(
    '此记录未提供决策详情',
  );
});

test('decision parameters remain plain text and wrap inside a narrow inspector', async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  const query = '<img src=x onerror="alert(1)">\n' + '合成升级参数_'.repeat(25);
  await connect(page, true, 'answered', false, query);
  await expect(page.locator('.answer-status')).toContainText('已生成答案');
  await page.locator('.timeline-row').filter({ hasText: 'Agent 决策' }).first().click();
  const details = page.getByRole('region', { name: '决策内容' });
  await expect(details).toContainText(query);
  await expect(details.locator('img')).toHaveCount(0);
  expect(await details.evaluate((element) => element.scrollWidth <= element.clientWidth)).toBe(
    true,
  );
});
