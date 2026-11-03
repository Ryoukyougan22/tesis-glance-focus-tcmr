import { defineConfig } from '@playwright/test';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
export default defineConfig({
  // Avoid OneDrive locks when Playwright cleans its transient diagnostic folder.
  outputDir: join(tmpdir(), 'videoqa-playwright'),
  testDir: './tests', timeout: 60000, workers: 1,
  use: { baseURL: process.env.VIDEOQA_FRONTEND || 'http://127.0.0.1:5173', channel: 'msedge', headless: true },
  webServer: { command: 'npm run dev', url: process.env.VIDEOQA_FRONTEND || 'http://127.0.0.1:5173', reuseExistingServer: true },
});
