import { defineConfig } from '@playwright/test';

for (const name of ['NO_PROXY', 'no_proxy']) {
  process.env[name] = [process.env[name], '127.0.0.1', 'localhost'].filter(Boolean).join(',');
}

export default defineConfig({
  testDir: './integration',
  timeout: 60_000,
  expect: { timeout: 20_000 },
  workers: 1,
  reporter: 'list',
  outputDir: 'test-results/persistence',
  use: {
    viewport: { width: 1440, height: 1000 },
    reducedMotion: 'reduce',
    trace: 'retain-on-failure',
  },
});
