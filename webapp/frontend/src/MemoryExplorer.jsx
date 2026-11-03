import React from 'react';
import { memoryName } from './format';
import MemoryDetail from './MemoryDetail';

export default function MemoryExplorer({caseId, state, inspect, grounding, groundingLoading, groundingError, onSeek, activeToken}) {
  const {result, analysis, inspected, status} = state;
  const pending = status === 'running';
  const selected = analysis?.memories.find(m => m.index === inspected);
  const standalone = result?.effect?.index === inspected ? result.effect : null;
  const detail = standalone || selected;
  return <section className="panel memories" aria-busy={pending}>
    <div className="section-heading"><span className="step">03</span><h2>Memorias internas</h2></div>
    <div className="memory-toolbar"><p>Selecciona E01–E10 para consultar su influencia y asociación visual.</p></div>
    <div className="memory-strip">{Array.from({length:10}, (_, i) => <button key={i} disabled={pending} aria-pressed={inspected === i+1} aria-label={`Inspeccionar ${memoryName(i+1)}`} className={inspected === i+1 ? 'inspected' : ''} onClick={() => inspect(i+1)}><b>{memoryName(i+1)}</b></button>)}</div>
    {groundingLoading && <p role="status">Cargando asociación visual precomputada…</p>}
    {groundingError && <p role="alert" className="error">{groundingError}</p>}
    {inspected && <MemoryDetail caseId={caseId} index={inspected} detail={pending ? undefined : detail} baseline={result?.baseline} grounding={grounding} technical={false} offline={false} pending={pending} activeToken={activeToken} tags={[]} onClose={() => inspect(null)} onSeek={onSeek}/>}
  </section>;
}
