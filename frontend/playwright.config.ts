import { defineConfig } from '@playwright/test';

const externalBaseURL = process.env.PLAYWRIGHT_BASE_URL;

// Playwright's readiness probe honors shell proxies; keep the local preview direct.
for (const name of ['NO_PROXY', 'no_proxy']) {
  process.env[name] = [process.env[name], '127.0.0.1', 'localhost'].filter(Boolean).join(',');
}

export default defineConfig({
  testDir: './e2e',
  timeout: 35_000,
  expect: { timeout: 20_000 },
  fullyParallel: true,
  workers: 2,
  reporter: 'list',
  use: {
    baseURL: externalBaseURL ?? 'http://127.0.0.1:4173',
    viewport: { width: 1440, height: 1000 },
    colorScheme: 'light',
    reducedMotion: 'reduce',
    trace: 'retain-on-failure',
  },
  webServer: externalBaseURL
    ? undefined
    : {
        command: 'npm run build && npm run preview -- --port 4173 --strictPort',
        url: 'http://127.0.0.1:4173/workbench/',
        reuseExistingServer: false,
      },
});
