import { expect, test } from '@playwright/test';
import { spawn, type ChildProcess } from 'node:child_process';
import { once } from 'node:events';
import path from 'node:path';

test('real backend: refresh, reconnect, supplement, pause, process restart and citations', async ({
  page,
  request,
}, info) => {
  const root = path.resolve('..');
  const python = path.join(
    root,
    '.venv',
    process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python',
  );
  const url = 'http://127.0.0.1:8013';
  const database = info.outputPath('synthetic.sqlite3');
  let server: ChildProcess | undefined;
  let logs = '';
  async function start() {
    server = spawn(
      python,
      ['scripts/serve_investigation_demo.py', '--port', '8013', '--db', database, '--delay', '1.2'],
      { cwd: root, windowsHide: true, stdio: 'pipe' },
    );
    server.stderr?.on('data', (data) => {
      logs += String(data);
    });
    await expect
      .poll(async () => {
        if (server?.exitCode !== null) throw new Error(logs);
        try {
          return (await request.get(`${url}/healthz`, { timeout: 500 })).status();
        } catch {
          return 0;
        }
      })
      .toBe(200);
  }
  async function stop() {
    if (!server || server.exitCode !== null) return;
    const exit = once(server, 'exit');
    server.kill('SIGKILL');
    await exit;
    server = undefined;
  }
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  try {
    await start();
    await page.addInitScript(() => localStorage.setItem('zhrag-run-mode', 'live'));
    await page.goto(`${url}/workbench/`);
    await expect(page.getByRole('combobox', { name: '调查模型' })).toContainText('synthetic-demo');
    await page.getByRole('textbox', { name: '调查问题' }).fill('星桥集群可以升级吗？');
    const creation = page.waitForResponse(
      (response) =>
        response.url().endsWith('/api/investigations') && response.request().method() === 'POST',
    );
    await page.getByRole('button', { name: '开始调查', exact: true }).click();
    const created = await (await creation).json();
    const runId = created.run_id;
    await page.reload();
    await expect(page.locator('.submitted-query')).toContainText('星桥集群');
    await expect(page.locator('.answer-status')).toContainText('等待补充信息');
    const before = await (await request.get(`${url}/api/investigations/${runId}`)).json();
    expect(before.response.usage).toMatchObject({ model_calls: 3, search_calls: 1, read_calls: 1 });
    await page.screenshot({ path: info.outputPath('waiting.png'), fullPage: true });
    await page.context().setOffline(true);
    await expect(page.getByText('进度连接已断开，正在重连。调查保留在服务端。')).toBeVisible();
    await page.context().setOffline(false);
    await expect(page.getByText('进度连接已断开，正在重连。调查保留在服务端。')).toHaveCount(0);
    await page.getByRole('textbox', { name: /请补充星桥集群/ }).fill('甲版升级到乙版');
    await page.getByRole('button', { name: '提交补充并继续' }).click();
    await page.getByRole('button', { name: '暂停调查' }).click();
    await expect(page.locator('.answer-status')).toContainText('已暂停');
    const paused = await (await request.get(`${url}/api/investigations/${runId}`)).json();
    expect(paused.response.sources[0].citation_id).toBe(1);
    await page.setViewportSize({ width: 390, height: 844 });
    await page.screenshot({ path: info.outputPath('paused-mobile.png'), fullPage: true });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(
      true,
    );
    await stop();
    await start();
    await page.reload();
    await expect(page.locator('.answer-status')).toContainText('已暂停');
    await page.getByRole('button', { name: '继续调查' }).click();
    await expect(page.locator('.answer-status')).toContainText('已生成答案');
    const final = await (await request.get(`${url}/api/investigations/${runId}`)).json();
    expect(final.response.usage).toMatchObject({ model_calls: 6, search_calls: 2, read_calls: 2 });
    expect(final.progress.slice(0, before.progress.length)).toEqual(before.progress);
    expect((await (await request.get(`${url}/api/investigations`)).json()).runs).toHaveLength(1);
    await page.getByRole('button', { name: '查看引用 1', exact: true }).first().click();
    await expect(
      page.getByText('【原创合成文档】星桥是虚构系统。', { exact: false }).first(),
    ).toBeVisible();
    await page.screenshot({ path: info.outputPath('answer-citation.png'), fullPage: true });
    await page.getByRole('button', { name: '关闭' }).last().click();
    await page.setViewportSize({ width: 1440, height: 1000 });
    await page.screenshot({ path: info.outputPath('completed-desktop.png'), fullPage: true });
    await page.getByRole('button', { name: '切换深色' }).click();
    await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
    await page.screenshot({
      path: info.outputPath('completed-dark.png'),
      fullPage: true,
      animations: 'disabled',
    });
    expect(errors).toEqual([]);
  } finally {
    await stop();
  }
});
