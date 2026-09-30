import { expect, test } from '@playwright/test';

for (const reducedMotion of ['no-preference', 'reduce'] as const) {
  for (const width of [888, 390]) {
    test(`select motion preserves reversal, selection and focus: ${width}, ${reducedMotion}`, async ({
      page,
    }, info) => {
      const errors: string[] = [];
      page.on('pageerror', (error) => errors.push(error.message));
      await page.route('**/api/**', (route) => route.abort());
      await page.setViewportSize({ width, height: 918 });
      await page.emulateMedia({ reducedMotion });
      await page.goto('/workbench/');
      if (width === 390) await page.getByRole('button', { name: '切换深色' }).click();

      for (const label of ['运行模式', '演示场景']) {
        const trigger = page.getByRole('combobox', { name: label, includeHidden: true });
        await trigger.click();
        const popup = page.locator('.select-content');
        const surface = await popup.elementHandle();
        const entrance = await popup.evaluate(async (el) => {
          const values: number[] = [];
          for (let i = 0; i < 6; i++) {
            await new Promise(requestAnimationFrame);
            values.push(Number(getComputedStyle(el).opacity));
          }
          return values;
        });
        if (reducedMotion === 'no-preference') {
          expect(entrance.some((v) => v > 0 && v < 1)).toBe(true);
        } else {
          expect(entrance.every((v) => v === 1)).toBe(true);
        }
        await expect(popup).toHaveCSS('opacity', '1');
        await expect(popup).toBeInViewport({ ratio: 1 });
        const position = await page.evaluate(() => scrollY);
        if (reducedMotion === 'no-preference') {
          // A real pointer click on the trigger must close, then reopen the
          // existing surface without waiting for the exit to finish.
          await trigger.click({ force: true });
          await page.waitForTimeout(20);
          await expect(popup).toHaveAttribute('inert', '');
          const closing = await popup.evaluate((el) => Number(getComputedStyle(el).opacity));
          expect(closing).toBeLessThan(1);
          await trigger.click({ force: true });
          expect(
            await surface!.evaluate((el) => el === document.querySelector('.select-content')),
          ).toBe(true);
          await expect(trigger).toHaveAttribute('aria-expanded', 'true');
          await expect(popup).toHaveCSS('opacity', '1');
          expect(await popup.evaluate((el) => el.contains(document.activeElement))).toBe(true);
        }
        await page.screenshot({ path: info.outputPath(`${label}-open.png`) });
        if (label === '运行模式') {
          await page.keyboard.press('End');
          await expect(page.getByRole('option', { name: '连接服务', exact: true })).toBeFocused();
          await page.keyboard.press('Escape');
          await expect(popup).toHaveCount(0);
          await expect(trigger).toHaveText('模拟演示');
        } else {
          await page.keyboard.press('Home');
          await page.keyboard.press('ArrowDown');
          await page.keyboard.press('Enter');
          await expect(popup).toHaveCount(0);
          await expect(trigger).toHaveText('需要补充信息');
        }
        await expect(trigger).toBeFocused();
        expect(await page.evaluate(() => scrollY)).toBe(position);
      }
      await page.getByRole('button', { name: '运行演示', exact: true }).click();
      await expect(page.getByRole('combobox', { name: '运行模式' })).toBeDisabled();
      await expect(page.getByRole('combobox', { name: '演示场景' })).toBeDisabled();
      await page.getByRole('button', { name: '停止', exact: true }).click();
      await expect(page.getByRole('combobox', { name: '演示场景' })).toBeEnabled();
      expect(errors).toEqual([]);
    });
  }
}

test('select cancels by outside click and responds to a changed motion preference', async ({
  page,
}) => {
  await page.route('**/api/**', (route) => route.abort());
  await page.emulateMedia({ reducedMotion: 'no-preference' });
  await page.goto('/workbench/');
  const trigger = page.getByRole('combobox', { name: '运行模式', includeHidden: true });
  await trigger.click();
  await expect(page.locator('.select-content')).toHaveCSS('opacity', '1');
  await page.mouse.click(100, 100);
  await expect(page.locator('.select-content')).toHaveCount(0);
  await expect(trigger).toBeFocused();
  await trigger.click();
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await expect(page.locator('.select-content')).toHaveCSS('opacity', '1');
  await page.keyboard.press('Escape');
  await expect(page.locator('.select-content')).toHaveCount(0, { timeout: 120 });
  await expect(trigger).toBeFocused();
});
