import { test, expect } from '@playwright/test';

async function ready(page) {
  await page.goto('/');
  await expect(page.getByText('Modelo conectado', {exact: false})).toBeVisible();
  await expect(page.locator('.case-state')).toHaveText('SIN EJECUTAR');
}

test('ten distinct videos, visual library, real inference and no automatic execution', async ({page}) => {
  await ready(page);
  await page.getByRole('button', {name:/BIBLIOTECA DE CASOS/}).click();
  await expect(page.locator('.case-card')).toHaveCount(10);
  await page.waitForFunction(() => [...document.querySelectorAll('.case-card img')].every(img => img.complete && img.naturalWidth > 0));
  await page.screenshot({path:'test-results/case-library.png',fullPage:true});
  await page.getByRole('button', {name:/BIBLIOTECA DE CASOS/}).click();
  const seen = new Set();
  for (let i=0;i<10;i++) {
    await expect(page.locator('.case-state')).toHaveText('SIN EJECUTAR');
    await page.waitForFunction(() => document.querySelector('video').readyState >= 1);
    seen.add(await page.locator('video').getAttribute('src'));
    await page.getByRole('button', {name:'Ejecutar Glance-Focus + TCMR'}).click();
    await expect(page.locator('.case-state')).toHaveText('LISTO');
    await expect(page.locator('.prediction strong')).toHaveText(/^[ABCD]$/);
    await expect(page.locator('.bar')).toHaveCount(4);
    if (i<9) await page.getByRole('button', {name:'Caso siguiente',exact:true}).click();
  }
  expect(seen.size).toBe(10);
});

test('case change during real request cannot publish stale results', async ({page}) => {
  await ready(page);
  await page.getByRole('button', {name:'Ejecutar Glance-Focus + TCMR'}).click();
  await page.getByRole('button', {name:'Caso siguiente',exact:true}).click();
  await expect(page.locator('.case-state')).toHaveText('SIN EJECUTAR');
  // A subsequent real inference serializes behind any still-running previous analysis.
  await page.getByRole('button', {name:'Ejecutar Glance-Focus + TCMR'}).click();
  await expect(page.locator('.case-state')).toHaveText('LISTO');
  await expect(page.locator('.ranking-chart')).toHaveCount(0);
  await expect(page.locator('.memory-detail')).toHaveCount(0);
});
