import { defineConfig } from 'vite';
import tailwindcss from '@tailwindcss/vite';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
const backend = process.env.VIDEOQA_BACKEND || 'http://127.0.0.1:8000';
export default defineConfig({cacheDir: join(tmpdir(), 'videoqa-vite-grounding'), plugins: [tailwindcss()], server: {proxy: {'/api': backend}}, preview: {proxy: {'/api': backend}}});
