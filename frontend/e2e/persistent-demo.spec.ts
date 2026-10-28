import { expect, test } from '@playwright/test';

for (const width of [1440, 390]) {
  test(`synthetic pause and resume keeps evidence (${width})`, async ({ page }, info) => {
    await page.setViewportSize({ width, height: width === 390 ? 844 : 1000 });
    await page.goto('/workbench/');
    await page.getByRole('combobox', { name: '演示场景' }).click();
    await page.getByRole('option', { name: '暂停与恢复', exact: true }).click();
    await page.getByRole('button', { name: '运行演示', exact: true }).click();
    await expect(page.locator('.answer-status')).toContainText('等待补充信息');
    await page
      .getByRole('textbox', { name: '请补充示例集群当前版本与目标版本。' })
      .fill('甲版到乙版');
    await page.getByRole('button', { name: '提交补充并继续' }).click();
    await page.getByRole('button', { name: '暂停调查' }).click();
    await expect(page.locator('.answer-status')).toContainText('已暂停');
    await page.screenshot({ path: info.outputPath('paused.png'), fullPage: true });
    await page.getByRole('button', { name: '继续调查' }).click();
    await expect(page.locator('.answer-status')).toContainText('已生成答案');
    await page.getByRole('button', { name: '查看引用 1', exact: true }).first().click();
    await expect(page.getByText('【原创合成演示文档】', { exact: false }).first()).toBeVisible();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(
      true,
    );
  });
}
