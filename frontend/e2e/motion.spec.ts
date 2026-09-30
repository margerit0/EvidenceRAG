import { expect, test } from '@playwright/test';

test.beforeEach(async ({ page }) => {
  await page.route('**/api/**', (route) => route.abort());
});

for (const reducedMotion of ['no-preference', 'reduce'] as const) {
  for (const viewport of [
    { width: 1440, height: 1000 },
    { width: 1366, height: 768 },
  ]) {
    test(`desktop reversal keeps graph and details continuous: ${viewport.width}, ${reducedMotion}`, async ({
      page,
    }, info) => {
      await page.setViewportSize(viewport);
      await page.emulateMedia({ reducedMotion });
      await page.goto('/workbench/');
      if (viewport.width === 1366) await page.getByRole('button', { name: '切换深色' }).click();
      await expect(page.locator('.flow-node')).toHaveCount(4);
      const samples = await page.evaluate(async () => {
        const track = document.querySelector('.inspector-slot')!;
        const pane = document.querySelector<HTMLElement>('.inspector-desktop')!;
        const toggle = document.querySelector<HTMLButtonElement>('.inspector-toggle')!;
        const node = document.querySelector('.flow-node');
        const samples: {
          t: number;
          width: number;
          inner: number;
          fits: boolean;
          sameNode: boolean;
          inert: boolean;
          open: boolean;
        }[] = [];
        const start = performance.now();
        const turns = [40, 130, 220, 330];
        let turn = 0;
        await new Promise<void>((resolve) => {
          const frame = (now: number) => {
            const t = now - start;
            if (turn < turns.length && t >= turns[turn]) {
              toggle.click();
              turn++;
            }
            const canvas = document.querySelector('.graph-canvas')!.getBoundingClientRect();
            const shapes = [
              ...document.querySelectorAll('.flow-node, .graph-edge .react-flow__edge-path'),
            ];
            samples.push({
              t,
              width: track.getBoundingClientRect().width,
              inner: pane.getBoundingClientRect().width,
              sameNode: node === document.querySelector('.flow-node'),
              inert: pane.inert,
              open: toggle.getAttribute('aria-expanded') === 'true',
              fits: shapes.every((shape) => {
                const r = shape.getBoundingClientRect();
                return (
                  r.height > 0 &&
                  (!shape.classList.contains('flow-node') || r.width > 0) &&
                  r.left >= canvas.left - 1 &&
                  r.right <= canvas.right + 1 &&
                  r.top >= canvas.top - 1 &&
                  r.bottom <= canvas.bottom + 1
                );
              }),
            });
            if (t < 950) requestAnimationFrame(frame);
            else resolve();
          };
          requestAnimationFrame(frame);
        });
        return samples;
      });
      await info.attach('reversal-geometry', {
        body: JSON.stringify(samples),
        contentType: 'application/json',
      });
      expect(samples.every((s) => s.fits && s.sameNode && (s.open || s.inert))).toBe(true);
      expect(
        Math.max(...samples.map((s) => s.inner)) - Math.min(...samples.map((s) => s.inner)),
      ).toBeLessThan(0.1);
      expect(samples.at(-1)!.width).toBe(0);
      if (reducedMotion === 'no-preference') {
        expect(samples.filter((s) => s.width > 5 && s.width < s.inner - 5).length).toBeGreaterThan(
          5,
        );
        for (let i = 1; i < samples.length; i++) {
          const a = samples[i - 1],
            b = samples[i];
          // Time-normalized upper bound detects a one-frame reset without making
          // any frame-rate claim or failing merely because a CI frame was slow.
          expect(Math.abs(b.width - a.width)).toBeLessThanOrEqual(
            a.inner * Math.min(1, ((b.t - a.t) / 420) * 10) + 2,
          );
        }
      } else {
        expect(samples.every((s) => s.width < 1 || Math.abs(s.width - s.inner) < 1)).toBe(true);
      }
      await expect(page.locator('.inspector-pane')).toBeHidden();
      await page.locator('.inspector-toggle').focus();
      await page.keyboard.press('Tab');
      expect(
        await page
          .locator('.inspector-desktop')
          .evaluate((el) => el.contains(document.activeElement)),
      ).toBe(false);
    });
  }
}

for (const viewport of [
  { width: 1024, height: 768 },
  { width: 390, height: 844 },
]) {
  test(`drawer reverses its live position and retains modal focus: ${viewport.width}`, async ({
    page,
  }) => {
    await page.setViewportSize(viewport);
    await page.emulateMedia({ reducedMotion: 'no-preference' });
    await page.goto('/workbench/');
    if (viewport.width === 390) await page.getByRole('button', { name: '切换深色' }).click();
    const toggle = page.locator('.inspector-toggle');
    await toggle.click();
    const drawer = page.locator('.inspector-drawer');
    await expect(drawer).toBeInViewport({ ratio: 1 });
    const surface = await drawer.elementHandle();
    const position = await page.evaluate(() => scrollY);
    await page.keyboard.press('Escape');
    await page.waitForTimeout(60);
    await expect(drawer).toHaveCount(1);
    await expect(drawer).toHaveAttribute('inert', '');
    const before = (await drawer.boundingBox())!.x;
    // The rail is fixed; force bypasses Playwright's animation stability wait so
    // this is an actual pointer click during the exit, not after it completes.
    await toggle.click({ force: true });
    const after = (await drawer.boundingBox())!.x;
    expect(after).toBeLessThan(viewport.width - 1);
    expect(Math.abs(after - before)).toBeLessThan(viewport.width * 0.5);
    await expect(toggle).toHaveAttribute('aria-expanded', 'true');
    await expect(drawer).toBeInViewport({ ratio: 1 });
    expect(
      await surface!.evaluate((el) => el === document.querySelector('.inspector-drawer')),
    ).toBe(true);
    await expect(drawer.getByRole('button', { name: '收起检查器' })).toBeFocused();
    await page.keyboard.press('Shift+Tab');
    expect(await drawer.evaluate((el) => el.contains(document.activeElement))).toBe(true);
    await page.mouse.click(8, 100);
    await expect(drawer).toHaveCount(0);
    await expect(toggle).toBeFocused();
    expect(await page.evaluate(() => scrollY)).toBe(position);
    // Changing the OS preference while open must also remove drawer travel.
    await toggle.click();
    await expect(drawer).toBeInViewport({ ratio: 1 });
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await page.keyboard.press('Escape');
    await expect(drawer).toHaveCount(0, { timeout: 250 });
    await expect(toggle).toBeFocused();
  });
}

test('history inspection preserves a manual graph zoom and execution signals stop at completion', async ({
  page,
}) => {
  await page.emulateMedia({ reducedMotion: 'no-preference' });
  await page.goto('/workbench/');
  await page.getByRole('button', { name: '运行演示', exact: true }).click();
  await expect(page.locator('.graph-edge-signal')).toHaveCount(1);
  await page.locator('.timeline-row').first().click();
  await expect(page.locator('.answer-status')).toContainText('已生成答案');
  await expect(page.locator('.flow-node.running, .graph-edge-signal')).toHaveCount(0);
  await page.getByRole('button', { name: '缩小流程图' }).click();
  const viewport = page.locator('.react-flow__viewport');
  const transform = await viewport.getAttribute('style');
  await page.locator('.timeline-row').filter({ hasText: '搜索文档' }).first().click();
  await expect(page.locator('.step-detail h2')).toHaveText('搜索文档 · 第 1 次');
  await expect(viewport).toHaveAttribute('style', transform!);
  await page.getByRole('button', { name: '下一次同类调用' }).click();
  await expect(viewport).toHaveAttribute('style', transform!);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.keyboard.press('Escape');
  await expect(page.locator('.inspector-drawer')).toHaveCount(0);
  await page.getByRole('button', { name: '适应画布', exact: true }).click();
  expect(
    await page.locator('.graph-canvas').evaluate((el) => {
      const canvas = el.getBoundingClientRect();
      const rail = document.querySelector('.inspector-rail')!.getBoundingClientRect();
      return [...el.querySelectorAll('.react-flow__edge-path')].every((edge) => {
        const r = edge.getBoundingClientRect();
        return (
          r.left >= canvas.left &&
          r.right < rail.left &&
          r.top >= canvas.top &&
          r.bottom <= canvas.bottom
        );
      });
    }),
  ).toBe(true);
});
