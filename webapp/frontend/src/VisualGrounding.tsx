import { es } from './localization/es';
import React, { useState } from 'react';
import { Image, Layers, MousePointer2 } from 'lucide-react';

import type { DistinctiveRegion, MemoryGrounding } from './types';
import { distinctiveRegions, visualInterpretation } from './visualInterpretation';

interface Props {
  memory: MemoryGrounding | undefined; technical: boolean;
  activeToken: number | null; onSeek: (region: DistinctiveRegion) => void;
}
function RegionImage({region}: {region: DistinctiveRegion}) {
  const [failed, setFailed] = useState(false);
  return failed || !region.frameUrl ? <span className="missing-frame">Fotograma no disponible</span> :
    <img src={region.frameUrl} onError={() => setFailed(true)} alt={`Región visual de referencia, token ${region.token}, aproximadamente ${region.approx_video_time_seconds.toFixed(2)} segundos`}/>;
}
export default function VisualGrounding({memory, technical, activeToken, onSeek}: Props) {

  if (!memory) return <div className="visual-grounding"><p>Análisis visual precomputado no disponible para esta memoria.</p></div>;
  const interpretation = visualInterpretation(memory.max_specificity_z);
  const regions = distinctiveRegions(memory);
  return <section className={`visual-grounding ${interpretation.kind}`} key={memory.memory}
    >
    <div className="flex items-center gap-2"><Image size={16} aria-hidden="true"/><h4>{es.association}</h4></div>
    <span className="data-source">Análisis visual precomputado</span>
    <h5>{interpretation.kind === 'distributed' ? <><Layers size={16}/> {es.distributed}</> : 'Regiones visuales distintivas'}</h5>
    <p>{interpretation.text}</p>
    {interpretation.kind === 'distributed' ? <p className="distributed-note">Ninguna de las regiones evaluadas mostró selectividad positiva para esta memoria frente a las demás. No se presentan fotogramas como evidencia distintiva.</p> : <>
      <div className="grounding-frames">{regions.map(region => <button key={region.token} className={activeToken === region.token ? 'active-frame' : ''}
        onClick={() => onSeek(region)} aria-label={`Ir a aproximadamente ${region.approx_video_time_seconds.toFixed(2)} segundos`} aria-pressed={activeToken === region.token}>
        <RegionImage region={region}/><span>≈ {region.approx_video_time_seconds.toFixed(2)} s</span><small>Sensibilidad: {region.sensitivity.toPrecision(4)} · Z: {region.specificity_z.toFixed(3)}</small>
      </button>)}</div>
      {regions.length === 0 && <p role="status">No hay fotogramas exportados disponibles para las regiones positivas.</p>}
      <p className="seek-hint"><MousePointer2 size={12}/> Clic en un fotograma → mover el video</p>
    </>}
    <div className="selectivity-metric"><span>Selectividad relativa</span><b>z max ≈ {memory.max_specificity_z.toFixed(2)}</b></div>
    <p className="visual-caveat">Los fotogramas indican sensibilidad visual relativa, no un significado semántico exclusivo de {memory.memory}. Los tiempos son aproximados, no intervalos temporales validados.</p>
    <details><summary>Sensibilidad y selectividad por región</summary><div className="table-wrap"><table><thead><tr><th>Región / token</th><th>≈ s</th><th>Selectividad Z</th><th>Sensibilidad</th></tr></thead><tbody>{memory.regions.map(r => <tr key={r.rank}><td>#{r.rank} / {r.token}</td><td>{r.approx_video_time_seconds.toFixed(2)}</td><td>{r.specificity_z.toPrecision(6)}</td><td>{r.sensitivity.toPrecision(6)}</td></tr>)}</tbody></table></div></details>
  </section>;
}
