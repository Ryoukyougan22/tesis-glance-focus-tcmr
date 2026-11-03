import { es } from './localization/es';
import { answerDisplay } from './localization/starCases.es';
import React, { useEffect, useRef } from 'react';
import { X, Activity } from 'lucide-react';
import type { CaseGrounding, DistinctiveRegion, MemoryEffect, Prediction } from './types';
import VisualGrounding from './VisualGrounding';
import { memoryName, pct, signed } from './format';

interface Props {caseId: string; index: number; detail: MemoryEffect | undefined;
  baseline: Pick<Prediction, 'prediction' | 'probability'> | undefined;
  grounding: CaseGrounding | null; technical: boolean; offline: boolean; pending: boolean;
  activeToken: number | null; tags: string[]; onClose: () => void;
  onSeek: (region: DistinctiveRegion) => void}

export default function MemoryDetail({caseId, index, detail, baseline, grounding, technical, offline, pending,
  activeToken, tags, onClose, onSeek}: Props) {
  const close = useRef<HTMLButtonElement>(null);
  const closeAction = useRef(onClose);
  closeAction.current = onClose;
  useEffect(() => {
    close.current?.focus({preventScroll:true});
    const handler = (event: KeyboardEvent) => {if (event.key === 'Escape') closeAction.current();};
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [index]);
  const name = memoryName(index);
  const memory = grounding?.memories.find(m => m.memory === name);
  return <div className="memory-detail visual-detail appear" role="dialog" aria-modal="false" aria-label={`Detalle ${name}`}>
    <button ref={close} className="close-inspector" onClick={onClose} aria-label="Cerrar detalle de memoria"><X size={18}/></button>
    <header><p className="eyebrow">{name} · MEMORIA INTERNA</p><h3>Representación latente interna</h3><div className="flex items-center gap-2">{detail?.rank && <span className="badge">Posición #{detail.rank} de 10</span>}{tags.map(tag => <span className="badge" key={tag}>{tag}</span>)}</div>{detail?.rank === 1 && <p>Memoria interna de mayor influencia</p>}</header>
    <section className="decision-influence"><h4 className="flex items-center gap-2"><Activity size={16}/>Análisis intervencional</h4>{detail && <span className="data-source">{offline ? 'Validación precomputada · no es una ejecución de esta sesión' : es.live}</span>}
      {detail && baseline ? <><div className="influence-metrics"><span>|Δp|<b>{detail.absolute_delta.toFixed(6)}</b></span><span>do({name}=0)<b>{baseline.prediction} → {detail.after.prediction}</b></span></div><p className={detail.flip ? 'memory-flip' : ''}>{detail.flip ? es.flip : es.noFlip}</p><p>Nueva respuesta: {detail.after.prediction}. {answerDisplay(caseId, detail.after.prediction)} · {pct(detail.after.probability)}</p><p>Δp firmado {signed(detail.delta)} · P({baseline.prediction}): {pct(baseline.probability)} → {pct(detail.original_answer_probability_after)}</p></> : <p role="status">{pending ? 'Calculando la condición original y las diez intervenciones…' : 'Ejecuta Glance-Focus + TCMR para consultar el efecto de esta memoria.'}</p>}
    </section>
    <VisualGrounding memory={memory} technical={technical} activeToken={activeToken} onSeek={onSeek}/>
  </div>;
}
