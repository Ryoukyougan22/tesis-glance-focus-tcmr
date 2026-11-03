import { es } from './localization/es';
export const letters = 'ABCD';
export const pct = n => `${(n * 100).toFixed(2)}%`;
export const signed = n => `${n >= 0 ? '+' : ''}${n.toFixed(6)}`;
export const memoryName = n => `E${String(n).padStart(2, '0')}`;
export const note = 'Las intervenciones permiten analizar la influencia de las representaciones internas sobre la decisión del modelo. No representan necesariamente relaciones causales físicas entre eventos del video.';

// Relative rank, not a universal effect-size threshold or statistical significance.
export const influence = rank => rank <= 3 ? 'Alta' : rank <= 7 ? 'Media' : 'Baja';
export function functionalRole(delta) {
  // Numerical display tolerance: one millionth of a probability, not a scientific cutoff.
  if (Math.abs(delta) < 1e-6) return {label: es.minimal, explanation: 'El cambio es menor que 0.000001 en probabilidad. Se conserva el valor sin redondear en modo técnico.'};
  return delta > 0
    ? {label: es.supports, explanation: 'Al apagar esta memoria, disminuye la probabilidad de la respuesta original.'}
    : {label: es.opposes, explanation: 'Al apagar esta memoria, aumenta la probabilidad de la respuesta original.'};
}
