import { es, states } from './localization/es';
import { answerDisplay } from './localization/starCases.es';
import React from 'react';
import { letters, pct, signed, memoryName } from './format';

function Prediction({label, value, caseId}) {
  return <div className="prediction"><p className="eyebrow">{label}</p><div><strong>{value.prediction}</strong><span>{pct(value.probability)}<small>Confianza</small></span></div><p>{answerDisplay(caseId, value.prediction)}</p><span className="badge">{value.correct === null ? 'Sin respuesta correcta' : value.correct ? 'Correcta ✓' : 'Incorrecta'}</span></div>;
}

export default function DecisionPanel({state, currentCase}) {
  const {status, result, operation} = state;
  const intervened = !!result?.selected;
  const pending = status === 'running';
  return <section className="panel decision-panel" aria-busy={pending}>
    <div className="section-heading"><span className="step">02</span><h2>{intervened ? 'Comparación de intervenciones' : 'Predicción del modelo'}</h2><span className={`case-state ${status}`}>{states[status]}</span></div>
    {pending ? <div className="execution" role="status"><span className="spinner"/><h3>{operation === 'analysis' ? 'Calculando influencia individual…' : operation === 'intervene' ? 'Ejecutando intervención…' : operation === 'restore' ? 'Restaurando memorias…' : 'Ejecutando inferencia…'}</h3><p>Glance-Focus + TCMR · Petición en curso</p><div className="skeleton title"/>{[0,1,2,3].map(i => <div className="skeleton" key={i}/>)}</div>
      : !result ? <div className="empty"><h3>Sin ejecución</h3><p>Ejecuta Glance-Focus + TCMR para consultar la predicción y las probabilidades.</p></div>
      : <div className="decision-content appear" key={intervened ? `intervened-${result.selected}` : 'baseline'}>
        <div className={`prediction-grid ${intervened ? 'comparison' : 'single'}`}>
          <Prediction label={intervened ? 'ANTES · TODAS ACTIVAS' : 'PREDICCIÓN DEL MODELO'} caseId={currentCase.id} value={result.baseline}/>
          {intervened && <Prediction label={`DESPUÉS · do(${memoryName(result.selected)}=0)`} caseId={currentCase.id} value={result.current}/>}
        </div>
        {intervened && <div className={`flip-banner ${result.effect.flip ? 'flipped' : ''}`}><strong>{result.baseline.prediction} → {result.current.prediction}</strong><span>{result.effect.flip ? es.flip : es.noFlip}</span></div>}
        {intervened && <div className="original-effect"><span>Probabilidad de la respuesta original · P({result.baseline.prediction})</span><b>{pct(result.baseline.probability)} → {pct(result.effect.original_answer_probability_after)}</b><div><span>Δp = {signed(result.effect.delta)}</span><span>|Δp| = {result.effect.absolute_delta.toFixed(6)}</span></div></div>}
        {intervened && <div className="legend"><span>● Antes</span><span>● Después</span></div>}
        <div className="probabilities">{currentCase.optionsDisplay.map((option, i) => <div className="prob-row" key={i}><div><b>{letters[i]}</b><span>{option}</span></div><div className="bar-row"><div className="track"><div className={`bar ${intervened ? 'before' : 'after'}`} style={{width: pct(result.baseline.probabilities[i])}}/></div><span>{pct(result.baseline.probabilities[i])}</span></div>{intervened && <div className="bar-row"><div className="track"><div className="bar after" style={{width: pct(result.current.probabilities[i])}}/></div><span>{pct(result.current.probabilities[i])}</span></div>}</div>)}</div>
        {!intervened && <p className="caption no-intervention">No se ha aplicado ninguna intervención · Todas las memorias activas.</p>}
      </div>}
  </section>;
}
