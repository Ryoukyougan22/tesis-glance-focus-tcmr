import React from 'react';
import { MapPin } from 'lucide-react';
import type { DistinctiveRegion, MemoryGrounding } from './types';
import { distinctiveRegions } from './visualInterpretation';

interface Props {duration: number; currentTime: number; memory: MemoryGrounding | undefined;
  activeToken: number | null; onSeek: (region: DistinctiveRegion) => void; onScrub: (time: number) => void}
export default function VisualTimeline({duration, currentTime, memory, activeToken, onSeek, onScrub}: Props) {
  const regions = distinctiveRegions(memory);
  return <div className="visual-timeline" aria-label="Línea de tiempo de asociación visual">
    <div className="timeline-label"><span className="flex items-center gap-2"><MapPin size={13}/>{memory?.memory || 'Memoria'} · Asociación visual relativa</span><span>{currentTime.toFixed(2)} / {duration.toFixed(2)} s</span></div>
    <div className="timeline-track"><input type="range" min={0} max={duration || 1} step="0.01" value={Math.min(currentTime,duration || 1)} disabled={!duration} aria-label="Posición del video" onChange={e => onScrub(Number(e.target.value))}/>
      <div className="timeline-markers">{duration > 0 && regions.map((region,i) => <button key={region.token} style={{left:`${Math.max(0,Math.min(100,region.approx_video_time_seconds / duration * 100))}%`, top:`${i*17}px`}} className={activeToken === region.token ? 'active-marker' : ''} disabled={region.approx_video_time_seconds > duration} onClick={() => onSeek(region)} title={`Región #${region.rank} · ≈ ${region.approx_video_time_seconds.toFixed(2)} s`} aria-label={`Marcador ${memory?.memory} región ${region.rank}, aproximadamente ${region.approx_video_time_seconds.toFixed(2)} segundos`}>●</button>)}</div>
    </div><div className="timeline-caption"><span>0 s</span><span>{memory?.max_specificity_z !== undefined && memory.max_specificity_z <= 0 ? 'Patrón distribuido · sin regiones distintivas' : '● Regiones aproximadas · alineamiento de features'}</span><span>{duration.toFixed(2)} s</span></div>
  </div>;
}
