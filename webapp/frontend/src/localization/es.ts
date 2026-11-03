// Presentation only. API keys, model inputs and scientific values stay unchanged.
export const es = {
  decision: 'Decisión del modelo', influence: 'Influencia en la decisión',
  association: 'Asociación visual', distributed: 'Patrón visual distribuido',
  highVisual: 'Alta selectividad visual', moderateVisual: 'Selectividad visual moderada',
  flip: 'Cambio de predicción', noFlip: 'Sin cambio de predicción',
  supports: 'Favorece la respuesta original', opposes: 'Se opone a la respuesta original',
  minimal: 'Efecto mínimo', truth: 'Respuesta correcta',
  offline: 'Validación precomputada', live: 'Inferencia real de esta sesión',
  networkError: 'No se pudo conectar con el servidor. Revisa la conexión e inténtalo de nuevo.',
  visualError: 'No se pudo cargar el análisis visual. Revisa la conexión e inténtalo de nuevo.',
};
export const states: Record<string, string> = {
  'not-run': 'SIN EJECUTAR', running: 'EJECUTANDO', ready: 'LISTO', intervened: 'INTERVENIDO',
};
export const categories: Record<string, string> = {
  Sequence: 'Secuencia', Interaction: 'Interacción', Prediction: 'Predicción', Feasibility: 'Factibilidad',
  Todas: 'Todos', Flip: 'Cambio', 'No Flip': 'Sin cambio',
};
export const operations: Record<string, string> = {
  inference: 'inferencia', intervention: 'intervención', analysis: 'análisis',
};
export const correlationLabels: Record<string, string> = {
  mean: 'Correlación media', median: 'Mediana de correlación', min: 'Correlación mínima', max: 'Correlación máxima',
};
const methods: Record<string, string> = {top: 'Mayor influencia', random: 'Aleatoria', bottom: 'Menor influencia'};
const actions: Record<string, string> = {
  top: 'Anular memoria de mayor influencia', random: 'Anular memoria aleatoria', bottom: 'Anular memoria de menor influencia',
};
export function methodLabel(key: string, technical = false) {
  return `${methods[key.toLowerCase()] || key}${technical ? ` (${key.toUpperCase()})` : ''}`;
}
export function interventionLabel(key: string, technical = false) {
  return `${actions[key]}${technical ? ` (${key.toUpperCase()})` : ''}`;
}
