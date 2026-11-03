import { es } from './localization/es';
import type { MemoryGrounding } from './types';

export function visualInterpretation(z: number) {
  if (z <= 0) return {label: es.distributed, kind: 'distributed',
    text: 'No se identificó una región visual claramente más distintiva para esta memoria que para las demás memorias internas.'};
  if (z <= 1.5) return {label: es.moderateVisual, kind: 'moderate',
    text: 'La memoria muestra una asociación visual moderadamente diferenciada.'};
  return {label: es.highVisual, kind: 'high',
    text: 'Esta memoria presenta regiones visuales relativamente distintivas.'};
}

export function distinctiveRegions(memory: MemoryGrounding | undefined) {
  if (!memory || memory.max_specificity_z <= 0) return [];
  // Use only existing positive-ranked regions, with actual exported images.
  return memory.regions.slice().sort((a,b) => a.rank - b.rank).slice(0,3)
    .filter(region => region.specificity_z > 0 && region.frameUrl !== null)
    .sort((a,b) => a.approx_video_time_seconds - b.approx_video_time_seconds);
}
