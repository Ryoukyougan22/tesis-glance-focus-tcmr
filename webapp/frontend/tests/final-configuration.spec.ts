import {test, expect} from '@playwright/test';
import {writeFileSync, existsSync, mkdirSync} from 'node:fs';
import {resolve, dirname} from 'node:path';
import {distinctiveRegions} from '../src/visualInterpretation';

test('final configuration: ten cases, 100 memories, exact visual source and isolation', async ({page}) => {
  test.setTimeout(240000);
  const output = resolve(process.env.VIDEOQA_VALIDATION_OUTPUT || '../../results/prototype_final_validation/browser.json');
  if (existsSync(output)) throw new Error('Choose a new browser evidence output.');
  mkdirSync(dirname(output), {recursive:true});
  const evidence: any = {cases:[], memories:0, frames_decoded:0, errors:[]};
  page.on('pageerror', error=>evidence.errors.push(String(error)));
  await page.goto('/');
  const cases = await (await page.request.get('/api/cases')).json();
  expect(cases).toHaveLength(10);
  for (const item of cases) {
    await page.getByRole('combobox',{name:'Caso STAR',exact:true}).selectOption(item.id);
    await expect(page.locator('.case-state')).toHaveText('SIN EJECUTAR');
    await expect(page.locator('.prediction')).toHaveCount(0);
    await expect(page.locator('.memory-detail')).toHaveCount(0);
    await page.waitForFunction(()=>document.querySelector('video')!.readyState>=2);
    const response=page.waitForResponse(r=>r.url().endsWith(`/cases/${item.id}/analysis`) && r.request().method()==='POST');
    await page.getByRole('button',{name:'Ejecutar Glance-Focus + TCMR'}).click();
    const analysis=await (await response).json();
    expect(analysis.reference_match).toBe(true);
    const grounding=await (await page.request.get(`/api/cases/${item.id}/grounding`)).json();
    expect(grounding.representation_stage).toBe('tcmr_refined');
    expect(grounding.checkpoint_sha256).toEqual(analysis.technical.checkpoint_sha256);
    const row:any={question_id:item.id,memories:[],state_cleared:true};
    for (const memory of grounding.memories) {
      await page.getByRole('button',{name:`Inspeccionar ${memory.memory}`,exact:true}).click();
      await expect(page.getByRole('dialog',{name:`Detalle ${memory.memory}`,exact:true})).toBeVisible();
      await expect(page.locator('.visual-grounding')).toContainText(`z max ≈ ${memory.max_specificity_z.toFixed(2)}`);
      const regions=distinctiveRegions(memory);
      await expect(page.locator('.grounding-frames img')).toHaveCount(regions.length);
      await page.waitForFunction(()=>Array.from(document.querySelectorAll<HTMLImageElement>('.grounding-frames img')).every(i=>i.complete&&i.naturalWidth>0));
      if (regions.length) {
        await page.locator('.grounding-frames img').first().click();
        await expect.poll(()=>page.locator('video').evaluate(v=>(v as HTMLVideoElement).currentTime)).toBeCloseTo(regions[0].approx_video_time_seconds,1);
      }
      row.memories.push(memory.memory);evidence.memories++;
      evidence.frames_decoded+=regions.length;
    }
    evidence.cases.push(row);
  }
  expect(evidence.memories).toBe(100);expect(evidence.errors).toEqual([]);
  writeFileSync(output,JSON.stringify(evidence,null,2),'utf8');
});
