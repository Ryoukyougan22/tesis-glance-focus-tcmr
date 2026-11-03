import {test, expect} from '@playwright/test';
import {readFileSync} from 'node:fs';
import {starCasesEs, presentCase, answerDisplay} from '../src/localization/starCases.es';
import type {DemoCase} from '../src/types';

test('ten fixed translations preserve original STAR inputs and option order', async ({page}) => {
  const originals = JSON.parse(readFileSync('../../results/demo_library/demo_cases_10_prepared.json','utf8'));
  await page.goto('/');
  const cases: DemoCase[] = await (await page.request.get('/api/cases')).json();
  expect(Object.keys(starCasesEs)).toHaveLength(10);
  for (const record of cases) {
    const original = originals.find((r: {question_id: string}) => r.question_id === record.id);
    expect(record.question).toBe(original.question);
    expect(record.options).toEqual(['a','b','c','d'].map(letter => original[`option_${letter}`]));
    const before = JSON.stringify(record);
    const display = presentCase(record);
    expect(JSON.stringify(record)).toBe(before);
    expect(display.questionOriginal).toBe(record.question);
    expect(display.optionsOriginal).toEqual(record.options);
    expect(display.optionsDisplay).toHaveLength(4);
    await page.getByRole('combobox',{name:'Caso STAR',exact:true}).selectOption(record.id);
    await expect(page.locator('.question h3')).toHaveText(display.questionDisplay);
    await expect(page.locator('.option>span')).toHaveText(display.optionsDisplay);
    expect(answerDisplay(record.id,record.correct_option)).toBe(display.optionsDisplay['ABCD'.indexOf(record.correct_option)]);
    await expect(page.locator('.case-state')).toHaveText('SIN EJECUTAR');
  }
});

test('Spanish presentation, methodology, real prediction and connection error', async ({page,context}) => {
  await page.setViewportSize({width:1440,height:900});
  await page.goto('/');
  await expect(page.locator('.question h3')).toHaveText(starCasesEs.Sequence_T3_4937.questionDisplay);
  const requestPromise = page.waitForRequest(r => r.url().endsWith('/analysis'));
  await page.getByRole('button',{name:'Ejecutar Glance-Focus + TCMR'}).click();
  expect((await requestPromise).postDataJSON()).toEqual({}); // Only the case ID selects the original server-side texts.
  await expect(page.locator('.case-state')).toHaveText('LISTO');
  await expect(page.locator('.prediction')).toContainText('Arrojó la escoba.');
  await page.getByRole('button',{name:'Inspeccionar E04',exact:true}).click();
  await page.locator('.visual-grounding summary').click();
  await expect(page.locator('.visual-grounding table')).toContainText('Sensibilidad');
  await page.screenshot({path:'test-results/es-tecnico.png',fullPage:true,animations:'disabled'});
  await page.locator('.visual-grounding summary').click();
  const visible = await page.locator('body').innerText();
  expect(visible).not.toMatch(/Internal Memor|Decision influence|Visual association|Prediction flip|Ground truth|CASE LIBRARY|Model Decision|Before|After|Distributed visual pattern|Supports original answer|High visual selectivity/);
  await page.getByRole('button',{name:'Cerrar detalle de memoria'}).click();
  await context.setOffline(true);
  await page.getByRole('button',{name:'Ejecutar Glance-Focus + TCMR'}).click();
  await expect(page.getByRole('alert').filter({hasText:'No se pudo conectar con el servidor'})).toBeVisible();
  await context.setOffline(false);
});
