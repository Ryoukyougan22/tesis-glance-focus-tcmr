import { categories } from './localization/es';
import { presentCase } from './localization/starCases.es';
import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { api, useWorkspace } from './useWorkspace';
import { letters, note, memoryName } from './format';
import DecisionPanel from './DecisionPanel';
import MemoryExplorer from './MemoryExplorer';
import CaseLibrary from './CaseLibrary';
import './styles.css';
import './workspace.css';
import './grounding.css';
import GroundedVideo from './GroundedVideo';


import { useGrounding } from './useGrounding';

function App() {
  const [cases, setCases] = useState([]);
  const [health, setHealth] = useState(null);
  const [connectionError, setConnectionError] = useState('');
  const [connecting, setConnecting] = useState(true);

  const [videoError, setVideoError] = useState(false);
  const {state, selectCase, inspect, run} = useWorkspace();
  const visual = useGrounding(state.caseId);
  const videoRef = useRef(null);
  const [activeRegion, setActiveRegion] = useState(null);
  const visualIndex = state.inspected;
  const visualMemory = visual.data?.memories.find(m => m.memory === memoryName(visualIndex));
  const activeToken = activeRegion && visualMemory && activeRegion.caseId === state.caseId && activeRegion.memory === visualMemory.memory
    ? activeRegion.token : null;
  function seekRegion(region) {
    const video = videoRef.current;
    if (!video || !Number.isFinite(video.duration) || region.approx_video_time_seconds > video.duration) return;
    video.currentTime = region.approx_video_time_seconds;
    setActiveRegion({caseId: state.caseId, memory: region.memory, token: region.token});
  }
  async function connect() {
    setConnecting(true); setConnectionError('');
    try {
      const [status, records] = await Promise.all([api('/health'), api('/cases')]);
      setHealth(status); setCases(records.map(presentCase)); selectCase(records[0]?.id || '');
    } catch (error) {setConnectionError(error.message); setHealth(null);}
    finally {setConnecting(false);}
  }
  useEffect(() => {connect();}, []);
  function choose(id) {selectCase(id); setVideoError(false); setActiveRegion(null);}
  const currentCase = cases.find(c => c.id === state.caseId);
  const busy = state.status === 'running';
  return <>
    <header className="topbar"><div><h1 className="academic-title">VideoQA – Análisis de representaciones internas</h1><p className="header-subtitle">Glance-Focus + TCMR</p></div><span className="connection"><b className={health ? 'dot live' : 'dot'}/>{health ? 'Modelo conectado' : connecting ? 'Conectando…' : 'Modelo sin conexión'}</span></header>
    <main className="research-workspace">
      {(connectionError || state.error) && <div role="alert" className="error">{connectionError || state.error}{!health && <button onClick={connect} disabled={connecting}>Reconectar</button>}</div>}
      {connecting && <p role="status" className="connection-loading"><span className="spinner"/> Consultando estado del servidor…</p>}
      {currentCase && <>
        <CaseLibrary cases={cases} current={currentCase} selectCase={choose}/>
        {state.result?.reference_match === false && <p role="alert" className="error">La inferencia actual difiere de la validación guardada. La asociación visual pertenece al experimento precomputado; su correspondencia con esta ejecución no está verificada.</p>}
        <div className="workspace" key={`case-${state.caseId}`}>
          <section className="panel video-panel"><div className="section-heading"><span className="step">01</span><h2>Video y pregunta</h2><span className="muted">{currentCase.video_id}</span></div><div className="case-content"><GroundedVideo caseId={state.caseId} videoRef={videoRef} memory={visualMemory} activeToken={activeToken} onSeek={seekRegion} onError={() => setVideoError(true)}/>{videoError && <p role="alert" className="error">No se pudo reproducir el video. Verifica el archivo MP4 y la conexión.</p>}<div className="question"><p className="eyebrow">PREGUNTA · {categories[currentCase.category]}</p><h3>{currentCase.questionDisplay}</h3><div className="options">{currentCase.optionsDisplay.map((option, i) => <div className={`option ${currentCase.correct_option === letters[i] ? 'truth' : ''}`} key={i}><b>{letters[i]}</b><span>{option}</span>{currentCase.correct_option === letters[i] && <small>Respuesta correcta</small>}</div>)}</div><button className="primary wide run-inference" disabled={busy} onClick={() => run('infer')}>{busy && state.operation === 'infer' ? <><span className="spinner"/> Ejecutando inferencia…</> : <>Ejecutar Glance-Focus + TCMR <span>→</span></>}</button></div></div></section>
          <DecisionPanel state={state} currentCase={currentCase}/>
        </div>
        <MemoryExplorer caseId={state.caseId} key={`memories-${state.caseId}`} state={state} run={run} inspect={inspect} technical={false} grounding={visual.data} groundingLoading={visual.loading} groundingError={visual.error} onSeek={seekRegion} activeToken={activeToken}/>
      </>}
      <aside className="method"><p>{note}</p></aside>
    </main>
  </>;
}

createRoot(document.getElementById('root')).render(<App/>);
