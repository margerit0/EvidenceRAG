import { expect, test } from '@playwright/test';
import { demoScript } from '../src/lib/demo';

test('desktop composition, dark theme persistence, and no horizontal overflow', async ({
  page,
}, info) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('/workbench/');
  await expect(page.getByRole('heading', { name: '从问题出发，让证据说话.' })).toBeVisible();
  await expect(page.locator('.flow-node')).toHaveCount(4);
  await page.screenshot({
    path: info.outputPath('light-desktop.png'),
    fullPage: true,
    animations: 'disabled',
  });
  await page.getByRole('button', { name: '切换深色' }).click();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  await expect(page.locator('.flow-node').first()).toHaveCSS('background-color', 'rgb(27, 33, 38)');
  await page.screenshot({
    path: info.outputPath('dark-desktop.png'),
    fullPage: true,
    animations: 'disabled',
  });
  await page.reload();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  expect(errors).toEqual([]);
});

test('repeated searches have separate details and citations open their evidence', async ({
  page,
}, info) => {
  await page.goto('/workbench/');
  await page.getByRole('button', { name: '运行演示', exact: true }).click();
  await expect(page.locator('.answer-status')).toContainText('已生成答案');
  await expect(page.locator('.timeline-action').filter({ hasText: '搜索文档' })).toHaveCount(2);
  await page.locator('.timeline-row').filter({ hasText: '搜索文档' }).last().click();
  await expect(page.locator('.step-detail h2')).toHaveText('搜索文档 · 第 2 次');
  await expect(page.locator('.step-detail')).toContainText('demo-6');
  await expect(page.locator('.flow-node').filter({ hasText: '搜索文档' })).toContainText(
    '最近第 2 次',
  );
  await page.locator('.timeline-row').filter({ hasText: '搜索文档' }).first().click();
  await expect(page.locator('.step-detail h2')).toHaveText('搜索文档 · 第 1 次');
  await expect(page.locator('.step-detail')).toContainText('demo-2');
  const searchNode = page.locator('.flow-node').filter({ hasText: '搜索文档' });
  await expect(searchNode).toHaveClass(/is-selected/);
  await expect(searchNode).toContainText('正在查看第 1 次');
  await expect(searchNode).toContainText('最近第 2 次');
  await expect(page.locator('.historical-call-note')).toContainText('“正在查看”与此处对应');
  const navigation = page.getByRole('navigation', { name: '同类调用导航' });
  await expect(navigation).toContainText('第 1 / 2 次');
  await expect(navigation.getByRole('button', { name: '上一次同类调用' })).toBeDisabled();
  await navigation.getByRole('button', { name: '下一次同类调用' }).click();
  await expect(page.locator('.step-detail h2')).toHaveText('搜索文档 · 第 2 次');
  await expect(searchNode).toContainText('正在查看第 2 次');
  await expect(page.locator('.historical-call-note')).toHaveCount(0);
  await expect(navigation).toContainText('第 2 / 2 次');
  await expect(navigation.getByRole('button', { name: '下一次同类调用' })).toBeDisabled();
  await expect(navigation.getByRole('button', { name: '跳到最新同类调用' })).toBeDisabled();
  await page.getByRole('button', { name: '收起检查器', exact: true }).click();
  await expect(page.locator('.inspector-pane')).toBeHidden();
  await page.getByRole('button', { name: '查看引用 2' }).click();
  await expect(page.locator('.inspector-pane')).toBeVisible();
  await expect(page.getByRole('button', { name: '查看引用 2' })).toHaveAttribute(
    'aria-pressed',
    'true',
  );
  await expect(page.getByRole('button', { name: '查看引用 1' })).toHaveAttribute(
    'aria-pressed',
    'false',
  );
  await expect(page.locator('.evidence-list [aria-pressed="true"]')).toContainText('回退条件');
  await expect(page.locator('.evidence-detail')).toContainText('回退条件');
  await expect(page.locator('.evidence-text')).toContainText('原创合成演示文档');
  await page.screenshot({
    path: info.outputPath('answer-and-evidence.png'),
    fullPage: true,
    animations: 'disabled',
  });
  await page.getByRole('button', { name: '新调查', exact: true }).click();
  await expect(
    page.getByRole('heading', { name: /一个问题，\s*一条清晰的证据路径。/ }),
  ).toBeVisible();
});

test('same-action navigation stays on history as calls arrive and supports adjacent calls', async ({
  page,
}, info) => {
  await page.goto('/workbench/');
  await page.getByRole('button', { name: '运行演示', exact: true }).click();
  const decisions = page.locator('.timeline-row').filter({ hasText: 'Agent 决策' });
  await expect(decisions.nth(1)).toBeVisible();
  await decisions.first().click();
  await expect(page.locator('.step-detail h2')).toHaveText('Agent 决策 · 第 1 次');
  await expect(page.locator('.answer-status')).toContainText('已生成答案');
  const navigation = page.getByRole('navigation', { name: '同类调用导航' });
  const previous = navigation.getByRole('button', { name: '上一次同类调用' });
  const next = navigation.getByRole('button', { name: '下一次同类调用' });
  const latest = navigation.getByRole('button', { name: '跳到最新同类调用' });
  await expect(navigation).toContainText('第 1 / 5 次');
  await expect(previous).toBeDisabled();
  await next.focus();
  await page.keyboard.press('Enter');
  await expect(page.locator('.step-detail h2')).toHaveText('Agent 决策 · 第 2 次');
  await expect(page.locator('.step-detail')).toContainText('demo-3');
  await expect(next).toBeFocused();
  await expect(page.locator('.timeline-row.selected')).toContainText('Agent 决策 · 第 2 次');
  await expect(page.locator('.flow-node.is-selected')).toContainText(
    '正在查看Agent 决策 · 第 2 次',
  );
  await page.screenshot({ path: info.outputPath('adjacent-decision.png') });
  await previous.click();
  await expect(navigation).toContainText('第 1 / 5 次');
  await latest.click();
  await expect(page.locator('.step-detail h2')).toHaveText('Agent 决策 · 第 5 次');
  await expect(next).toBeDisabled();
  await expect(latest).toBeDisabled();
  await previous.click();
  await expect(page.locator('.step-detail h2')).toHaveText('Agent 决策 · 第 4 次');
  await expect(navigation).toContainText('第 4 / 5 次');
  // These share the finish graph node, but are distinct action types.
  for (const action of ['校验答案', '结束调查']) {
    await page.locator('.timeline-row').filter({ hasText: action }).click();
    await expect(page.locator('.step-detail h2')).toHaveText(`${action} · 第 1 次`);
    await expect(navigation).toContainText('第 1 / 1 次');
    await expect(previous).toBeDisabled();
    await expect(next).toBeDisabled();
    await expect(latest).toBeDisabled();
  }
  await decisions.first().click();
  await page.setViewportSize({ width: 390, height: 844 });
  await navigation.scrollIntoViewIfNeeded();
  await next.click();
  await expect(page.locator('.step-detail h2')).toHaveText('Agent 决策 · 第 2 次');
  await expect(navigation).toBeInViewport({ ratio: 1 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({ path: info.outputPath('mobile-adjacent-decision.png') });
});

for (const scenario of [
  { name: '需要补充信息', result: '需要补充信息' },
  { name: '证据不足', result: '证据不足' },
  { name: '调用超时', result: '模型调用超时' },
  { name: '预算耗尽', result: '已达预算上限' },
]) {
  test(`terminal state: ${scenario.name}`, async ({ page }) => {
    await page.goto('/workbench/');
    await page.getByRole('combobox', { name: '演示场景' }).click();
    await page.getByRole('option', { name: scenario.name, exact: true }).click();
    await page.getByRole('button', { name: '运行演示', exact: true }).click();
    await expect(page.locator('.answer-status')).toContainText(scenario.result);
    await expect(page.locator('.answer-body')).toHaveCount(0);
    await page.getByRole('button', { name: '展开检查器', exact: true }).click();
    await expect(page.locator('.inspector-outcome')).toContainText(scenario.result);
    if (scenario.name !== '需要补充信息') {
      await page.getByRole('button', { name: '查看已读取证据' }).click();
      await expect(page.locator('.evidence-text')).toContainText('原创合成演示文档');
    }
    await expect(page.getByRole('button', { name: '运行演示', exact: true })).toBeEnabled();
  });
}

test('stopping a demo retains steps and permits a new run', async ({ page }) => {
  await page.goto('/workbench/');
  await page.getByRole('button', { name: '运行演示', exact: true }).click();
  await expect(page.locator('.timeline-row').first()).toBeVisible();
  await page.getByRole('button', { name: '停止', exact: true }).click();
  await expect(page.locator('.answer-status')).toContainText('调查已停止');
  await expect(page.locator('.timeline-row').first()).toBeVisible();
  await expect(page.getByRole('button', { name: '运行演示', exact: true })).toBeEnabled();
  await page.getByRole('button', { name: '运行历史' }).click();
  await expect(page.locator('.history-item')).toHaveCount(1);
  await page.locator('.history-item').click();
  await expect(page.getByRole('dialog')).toHaveCount(0);
});

test('mobile citation opens evidence and restores focus and reading position', async ({
  page,
}, info) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/workbench/');
  await page.getByRole('button', { name: '运行演示', exact: true }).click();
  await expect(page.locator('.current-action')).toContainText('执行中');
  const progress = page.locator('.mobile-progress');
  await expect(progress).toBeInViewport({ ratio: 1 });
  await expect(progress).toContainText('执行中');
  await expect(page.locator('.working-copy')).toContainText('下方');
  await page.getByRole('button', { name: '查看流程', exact: true }).click();
  await expect(page.getByRole('region', { name: '执行流程', exact: true })).toBeFocused();
  await expect(progress).toBeInViewport({ ratio: 1 });
  await expect(page.locator('.current-action')).toBeInViewport({ ratio: 1 });
  await expect(page.locator('.answer-status')).toContainText('已生成答案');
  const citation = page.getByRole('button', { name: '查看引用 2', exact: true });
  await citation.scrollIntoViewIfNeeded();
  const target = await citation.boundingBox();
  expect(target!.width).toBeGreaterThanOrEqual(40);
  expect(target!.height).toBeGreaterThanOrEqual(40);
  const position = await page.evaluate(() => scrollY);
  await citation.click();
  const dialog = page.getByRole('dialog', { name: '引用证据' });
  await expect(dialog).toBeVisible();
  await expect(dialog).toContainText('升级演练 / 回退条件');
  await page.screenshot({ path: info.outputPath('mobile-evidence-drawer.png') });
  await page.keyboard.press('Escape');
  await expect(dialog).toHaveCount(0);
  await expect(citation).toBeFocused();
  expect(await page.evaluate(() => scrollY)).toBe(position);
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.getByRole('button', { name: '展开检查器', exact: true }).click();
  await page.getByRole('button', { name: '展开阅读' }).click();
  await expect(dialog).toContainText('升级演练 / 回退条件');
});

test('short desktop keeps controls visible and selected graph labels clear', async ({
  page,
}, info) => {
  await page.setViewportSize({ width: 1366, height: 768 });
  await page.goto('/workbench/');
  await page.getByRole('button', { name: '展开检查器', exact: true }).click();
  await expect(page.getByRole('textbox', { name: '调查问题' })).toBeInViewport({ ratio: 1 });
  await expect(page.getByRole('button', { name: '运行演示', exact: true })).toBeInViewport({
    ratio: 1,
  });
  await page.getByRole('button', { name: '运行演示', exact: true }).click();
  await expect(page.getByRole('button', { name: '停止', exact: true })).toBeInViewport({
    ratio: 1,
  });
  await expect(page.locator('.answer-status')).toContainText('已生成答案');
  const before = await page
    .locator('.flow-node')
    .evaluateAll((nodes) => nodes.map((node) => node.getBoundingClientRect().height));
  await page.locator('.timeline-row').filter({ hasText: '搜索文档' }).first().click();
  await expect(page.locator('.step-detail h2')).toHaveText('搜索文档 · 第 1 次');
  const after = await page
    .locator('.flow-node')
    .evaluateAll((nodes) => nodes.map((node) => node.getBoundingClientRect().height));
  expect(after).toEqual(before);
  const geometry = await page.evaluate(() => {
    const caption = document.querySelector('.graph-caption')!.getBoundingClientRect();
    const canvas = document.querySelector('.graph-canvas')!.getBoundingClientRect();
    const nodes = [...document.querySelectorAll<HTMLElement>('.flow-node')];
    const rows = [...document.querySelectorAll('.timeline-row')];
    return {
      clearLabels: nodes.every((node) => {
        const rect = node.getBoundingClientRect();
        const title = node.querySelector('.node-heading')!;
        const visibleFontSize =
          (parseFloat(getComputedStyle(title).fontSize) * rect.width) / node.offsetWidth;
        return rect.bottom < caption.top && rect.top >= canvas.top && visibleFontSize >= 13;
      }),
      timelineHeight: document.querySelector('.timeline')!.clientHeight,
      threeRows: rows
        .slice(0, 3)
        .reduce((height, row) => height + row.getBoundingClientRect().height, 0),
    };
  });
  expect(geometry.clearLabels).toBe(true);
  expect(geometry.timelineHeight).toBeGreaterThanOrEqual(geometry.threeRows);
  await page.screenshot({ path: info.outputPath('laptop-selected-call.png') });
});

test('inspector collapse frees space and preserves the selected call and tab', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'no-preference' });
  await page.goto('/workbench/');
  const expectGraphFits = async () => {
    await expect
      .poll(() =>
        page.evaluate(() => {
          const canvas = document.querySelector('.graph-canvas')!.getBoundingClientRect();
          const nodes = [...document.querySelectorAll('.flow-node')];
          return (
            nodes.length === 4 &&
            nodes.every((node) => {
              const rect = node.getBoundingClientRect();
              return (
                rect.width > 0 &&
                rect.height > 0 &&
                rect.left >= canvas.left &&
                rect.right <= canvas.right &&
                rect.top >= canvas.top &&
                rect.bottom <= canvas.bottom
              );
            })
          );
        }),
      )
      .toBe(true);
  };
  const pane = page.locator('.inspector-pane');
  await expect(pane).toBeHidden();
  await expect(page.getByRole('button', { name: '展开检查器', exact: true })).toHaveAttribute(
    'aria-expanded',
    'false',
  );
  const collapsedWidth = (await page.locator('.execution-pane').boundingBox())!.width;
  await expectGraphFits();
  await page.getByRole('button', { name: '展开检查器', exact: true }).click();
  await expectGraphFits();
  await page.getByRole('button', { name: '收起检查器', exact: true }).click();
  await expectGraphFits();
  await page.getByRole('button', { name: '运行演示', exact: true }).click();
  await expect(page.locator('.answer-status')).toContainText('已生成答案');
  await expect(pane).toBeHidden();
  await page.locator('.timeline-row').filter({ hasText: 'Agent 决策' }).first().click();
  await expect(pane).toBeVisible();
  await expectGraphFits();
  expect((await page.locator('.execution-pane').boundingBox())!.width).toBeLessThan(
    collapsedWidth - 100,
  );
  await page.getByRole('button', { name: '下一次同类调用' }).click();
  await page.getByRole('tab', { name: '运行', exact: true }).click();
  await page.getByRole('button', { name: '收起检查器', exact: true }).click();
  await expect(pane).toBeHidden();
  await expectGraphFits();
  expect((await page.locator('.execution-pane').boundingBox())!.width).toBeCloseTo(
    collapsedWidth,
    0,
  );
  const expand = page.getByRole('button', { name: '展开检查器', exact: true });
  await expect(expand).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(page.getByRole('tab', { name: '运行', exact: true })).toHaveAttribute(
    'data-state',
    'active',
  );
  await page.getByRole('tab', { name: '步骤', exact: true }).click();
  await expect(page.locator('.step-detail h2')).toHaveText('Agent 决策 · 第 2 次');
  await page.setViewportSize({ width: 1366, height: 768 });
  await expectGraphFits();
});

test('narrow inspector opens at the right edge and restores focus without moving the page', async ({
  page,
}, info) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/workbench/');
  const expand = page.getByRole('button', { name: '展开检查器', exact: true });
  await expect(expand).toBeInViewport({ ratio: 1 });
  await page.getByRole('region', { name: '执行流程', exact: true }).scrollIntoViewIfNeeded();
  const position = await page.evaluate(() => scrollY);
  await expand.click();
  const drawer = page.getByRole('dialog', { name: '检查器', exact: true });
  await expect(drawer).toBeInViewport({ ratio: 1 });
  const bounds = (await drawer.boundingBox())!;
  expect(bounds.x + bounds.width).toBe(390);
  await expect(drawer.getByRole('button', { name: '收起检查器', exact: true })).toBeFocused();
  await page.getByRole('tab', { name: '运行', exact: true }).click();
  await page.screenshot({ path: info.outputPath('mobile-inspector-open.png') });
  await page.keyboard.press('Escape');
  await expect(drawer).toHaveCount(0);
  await expect(expand).toBeFocused();
  expect(await page.evaluate(() => scrollY)).toBe(position);
  await expand.click();
  await expect(page.getByRole('tab', { name: '运行', exact: true })).toHaveAttribute(
    'data-state',
    'active',
  );
  await drawer.getByRole('button', { name: '收起检查器', exact: true }).click();
  await expect(expand).toBeFocused();
  await page.setViewportSize({ width: 1024, height: 768 });
  await expand.click();
  await expect(drawer).toBeInViewport({ ratio: 1 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});

test('narrow viewport reflows into continuous sections', async ({ page }, info) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/workbench/');
  await expect(page.getByRole('button', { name: '运行演示', exact: true })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({
    path: info.outputPath('mobile-light.png'),
    fullPage: true,
    animations: 'disabled',
  });
  await page.getByRole('button', { name: '切换深色' }).click();
  await page.screenshot({
    path: info.outputPath('mobile-dark.png'),
    fullPage: true,
    animations: 'disabled',
  });
});

test('connected mode uses one POST, renders streamed results and treats content as text', async ({
  page,
}) => {
  const script = demoScript('answered', 'synthetic-service');
  let requests = 0;
  script.result.blocks[0].text = '<img src=x onerror="alert(1)"> 合成响应';
  await page.route('**/api/capabilities', (route) =>
    route.fulfill({
      json: { agent_enabled: true, agent_profile: 'synthetic', agent_streaming: true },
    }),
  );
  await page.route('**/api/investigate/stream', (route) => {
    requests++;
    return route.fulfill({
      contentType: 'text/event-stream',
      body:
        script.events
          .map((event) => `event: progress\ndata: ${JSON.stringify(event)}\n\n`)
          .join('') +
        `event: result\ndata: ${JSON.stringify({ run_id: 'synthetic-service', response: script.result })}\n\n`,
    });
  });
  await page.goto('/workbench/');
  await page.getByRole('combobox', { name: '运行模式' }).click();
  await page.getByRole('option', { name: '连接服务', exact: true }).click();
  await expect(page.locator('.connection-note')).toContainText('服务已连接');
  await page.getByRole('textbox', { name: '调查问题' }).fill('合成服务问题');
  await page.getByRole('button', { name: '开始调查', exact: true }).click();
  await expect(page.locator('.answer-status')).toContainText('已生成答案');
  await expect(page.locator('.answer-body')).toContainText('<img src=x');
  await expect(page.locator('.answer-body img')).toHaveCount(0);
  expect(requests).toBe(1);
});
