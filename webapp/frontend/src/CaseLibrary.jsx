import { categories } from './localization/es';
import React, { useState } from 'react';

export default function CaseLibrary({cases, current, selectCase}) {
  const [open, setOpen] = useState(false);
  const [filter, setFilter] = useState('Todas');
  const index = cases.findIndex(c => c.id === current.id);
  function choose(id) { selectCase(id); setOpen(false); }
  return <section className="case-library">
    <div className="case-bar"><button className="library-toggle" aria-expanded={open} onClick={() => setOpen(!open)}>▦ BIBLIOTECA DE CASOS <span>{cases.length} casos STAR</span> {open ? '▴' : '▾'}</button><label className="case-select"><span className="sr-only">Caso STAR</span><select value={current.id} onChange={e => choose(e.target.value)}>{cases.map(c => <option key={c.id} value={c.id}>{c.video_id} · {c.id}</option>)}</select></label><span className="badge">{categories[current.category]}</span><div className="case-navigation"><button disabled={index <= 0} onClick={() => choose(cases[index - 1].id)} aria-label="Caso anterior">← Anterior</button><span>{index + 1} / {cases.length}</span><button disabled={index >= cases.length - 1} onClick={() => choose(cases[index + 1].id)} aria-label="Caso siguiente">Siguiente →</button></div></div>
    {open && <div className="library-content appear"><div className="library-filters">{['Todas', 'Sequence', 'Interaction', 'Prediction', 'Feasibility'].map(category => <button aria-pressed={filter === category} onClick={() => setFilter(category)} key={category}>{categories[category]}</button>)}</div><div className="case-cards">{cases.filter(c => filter === 'Todas' || c.category === filter || (filter === 'Flip' && c.reference?.top_flip) || (filter === 'No Flip' && c.reference && !c.reference.top_flip)).map(c => <button className={`case-card ${c.id === current.id ? 'current' : ''}`} key={c.id} onClick={() => choose(c.id)}><img loading="lazy" src={`/api/cases/${encodeURIComponent(c.id)}/thumbnail`} alt={`Fotograma del video ${c.video_id}`}/><div><span className="badge">{categories[c.category]}</span><b>{c.video_id}</b><small>{c.id}</small><p>{c.questionDisplay}</p><small>Respuesta correcta: {c.correct_option} · {c.video_id}</small></div></button>)}</div><p className="caption">Casos STAR reales con videos y features S3D oficiales. Seleccionar un caso no ejecuta inferencia.</p></div>}
  </section>;
}
