import { useEffect, useState } from 'react';
import { es } from './localization/es';
import type { CaseGrounding } from './types';

const cache = new Map<string, CaseGrounding>();
export function useGrounding(caseId: string) {
  const [data, setData] = useState<CaseGrounding | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  useEffect(() => {
    const controller = new AbortController();
    setError(''); setData(null);
    if (!caseId) {setLoading(false); return;}
    const saved = cache.get(caseId);
    if (saved) {setData(saved); setLoading(false); return;}
    setLoading(true);
    fetch(`/api/cases/${encodeURIComponent(caseId)}/grounding`, {signal: controller.signal})
      .then(async response => {
        if (!response.ok) throw new Error('El análisis visual precomputado no está disponible. La inferencia sigue disponible.');
        const payload: CaseGrounding = await response.json();
        if (payload.case_id !== caseId) throw new Error('El análisis visual recibido no corresponde al caso seleccionado.');
        if (!controller.signal.aborted) {cache.set(caseId, payload); setData(payload);}
      })
      .catch((cause: unknown) => {if (!controller.signal.aborted) setError(cause instanceof Error && !(cause instanceof TypeError) ? cause.message : es.visualError);})
      .finally(() => {if (!controller.signal.aborted) setLoading(false);});
    return () => controller.abort();
  }, [caseId]);
  return {data: data?.case_id === caseId ? data : null, error, loading};
}
