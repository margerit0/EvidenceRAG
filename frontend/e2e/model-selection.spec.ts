import { expect, test } from '@playwright/test';
import { demoScript } from '../src/lib/demo';

const models = [
  { id: 'deepseek-v4.1-flash', name: 'DeepSeek V4.1 Flash' },
  { id: 'z-ai/glm-5.3', name: 'GLM 5.3' },
];

for (const width of [1440, 390]) {
  for (const theme of ['light', 'dark'] as const) {
    test(`model selection routes each run and preserves history (${width}, ${theme})`, async ({
      page,
    }, info) => {
      await page.setViewportSize({ width, height: width === 390 ? 844 : 1000 });
      await page.addInitScript((theme) => localStorage.setItem('zhrag-theme', theme), theme);
      await page.route('**/api/**', (route) => route.abort());
      await page.route('**/api/capabilities', (route) =>
        route.fulfill({
          json: {
            agent_enabled: true,
            agent_profile: 'default-profile',
            agent_streaming: true,
            agent_models: models,
            default_agent_model: models[0].id,
          },
        }),
      );
      const requests: string[] = [];
      let release: (() => void) | undefined;
      await page.route('**/api/investigate/stream', async (route) => {
        const model = route.request().postDataJSON().model as string;
        requests.push(model);
        await new Promise<void>((resolve) => {
          release = resolve;
        });
        const runId = `synthetic-model-${requests.length}`;
        const script = demoScript('answered', runId);
        await route.fulfill({
          contentType: 'text/event-stream',
          body:
            script.events
              .map((event) => `event: progress\ndata: ${JSON.stringify(event)}\n\n`)
              .join('') +
            `event: result\ndata: ${JSON.stringify({ run_id: runId, response: { ...script.result, model, agent_profile: model } })}\n\n`,
        });
      });
      await page.goto('/workbench/');
      await expect(page.getByRole('combobox', { name: '调查模型' })).toHaveCount(0);
      await page.getByRole('combobox', { name: '运行模式' }).click();
      await page.getByRole('option', { name: '连接服务', exact: true }).click();
      const selector = page.getByRole('combobox', { name: '调查模型' });
      await expect(selector).toContainText('DeepSeek');
      await expect(
        page.locator('.page-heading').getByRole('combobox', { name: '调查模型' }),
      ).toHaveCount(0);
      await expect(
        page.locator('.composer-actions').getByRole('combobox', { name: '调查模型' }),
      ).toBeVisible();
      const modelBox = await selector.boundingBox();
      const sendBox = await page
        .getByRole('button', { name: '开始调查', exact: true })
        .boundingBox();
      expect(modelBox!.x + modelBox!.width).toBeLessThanOrEqual(sendBox!.x);
      expect(
        Math.abs(modelBox!.y + modelBox!.height / 2 - sendBox!.y - sendBox!.height / 2),
      ).toBeLessThan(3);
      await selector.click();
      await expect(page.getByRole('option', { name: 'GLM 5.3', exact: true })).toBeVisible();
      const menu = await page.getByRole('listbox').boundingBox();
      expect(menu).not.toBeNull();
      expect(menu!.x).toBeGreaterThanOrEqual(0);
      expect(menu!.x + menu!.width).toBeLessThanOrEqual(width);
      expect(menu!.y + menu!.height).toBeLessThanOrEqual(modelBox!.y);
      await page.screenshot({ path: info.outputPath('model-menu.png') });
      await page.getByRole('option', { name: 'GLM 5.3', exact: true }).click();
      await expect(selector).toBeFocused();
      await page.getByRole('textbox', { name: '调查问题' }).fill('第一次合成模型调查');
      await page.getByRole('button', { name: '开始调查', exact: true }).click();
      await expect.poll(() => requests.length).toBe(1);
      await expect(selector).toBeDisabled();
      await expect(page.locator('.data-tag')).toContainText('GLM 5.3');
      release!();
      await expect(page.locator('.answer-status')).toContainText('已生成答案');
      await selector.click();
      await page.getByRole('option', { name: 'DeepSeek V4.1 Flash', exact: true }).click();
      await expect(page.locator('.data-tag')).toContainText('GLM 5.3');
      await page.getByRole('textbox', { name: '调查问题' }).fill('第二次合成模型调查');
      await page.getByRole('button', { name: '开始调查', exact: true }).click();
      await expect.poll(() => requests.length).toBe(2);
      release!();
      await expect(page.locator('.answer-status')).toContainText('已生成答案');
      await expect(page.locator('.data-tag')).toContainText('DeepSeek V4.1 Flash');
      await page.getByRole('button', { name: '运行历史' }).click();
      const first = page.getByRole('button').filter({ hasText: '第一次合成模型调查' });
      await expect(first).toContainText('GLM 5.3');
      await first.click();
      await expect(page.locator('.data-tag')).toContainText('GLM 5.3');
      expect(requests).toEqual([models[1].id, models[0].id]);
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(
        true,
      );
      await page.screenshot({ path: info.outputPath('model-history.png'), fullPage: true });
      await selector.click();
      await page.getByRole('option', { name: 'GLM 5.3', exact: true }).click();
      await page.reload();
      await page.getByRole('combobox', { name: '运行模式' }).click();
      await page.getByRole('option', { name: '连接服务', exact: true }).click();
      await expect(page.getByRole('combobox', { name: '调查模型' })).toContainText('GLM 5.3');
      expect(requests).toHaveLength(2);
    });
  }
}

test('unavailable saved model falls back to advertised default without submitting', async ({
  page,
}) => {
  const posts: string[] = [];
  await page.addInitScript(() => localStorage.setItem('zhrag-agent-model', 'retired-model'));
  await page.route('**/api/**', (route) => {
    posts.push(route.request().method());
    return route.abort();
  });
  await page.route('**/api/capabilities', (route) =>
    route.fulfill({
      json: {
        agent_enabled: true,
        agent_profile: 'default',
        agent_models: models,
        default_agent_model: models[0].id,
      },
    }),
  );
  await page.goto('/workbench/');
  await page.getByRole('combobox', { name: '运行模式' }).click();
  await page.getByRole('option', { name: '连接服务', exact: true }).click();
  await expect(page.getByRole('combobox', { name: '调查模型' })).toContainText('DeepSeek');
  expect(posts).toEqual([]);
});
