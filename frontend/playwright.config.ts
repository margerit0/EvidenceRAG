import { defineConfig } from '@playwright/test';

const externalBaseURL = process.env.PLAYWRIGHT_BASE_URL;

export default defineConfig({
  testDir: './e2e',
  timeout: 35_000,
  expect: { timeout: 20_000 },
  fullyParallel: true,
  workers: 2,
  reporter: 'list',
  use: {
    baseURL: externalBaseURL ?? 'http://127.0.0.1:5173',
    channel: process.env.CI ? undefined : 'chrome',
    viewport: { width: 1440, height: 1000 },
    colorScheme: 'light',
    reducedMotion: 'reduce',
    trace: 'retain-on-failure',
  },
  webServer: externalBaseURL
    ? undefined
    : {
        command: 'npm run dev',
        url: 'http://127.0.0.1:5173/workbench/',
        reuseExistingServer: !process.env.CI,
      },
});
