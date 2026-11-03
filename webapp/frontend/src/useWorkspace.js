import { useReducer, useRef } from 'react';
import { es } from './localization/es';

export async function api(path, body, signal) {
  const response = await fetch(`/api${path}`, {
    signal, ...(body === undefined ? {} : {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body),
    }),
  }).catch(error => {
    if (error.name === 'AbortError') throw error;
    throw new Error(es.networkError);
  });
  if (!response.ok) {
    const data = await response.json().catch(() => null);
    throw new Error(typeof data?.detail === 'string' ? data.detail : `La solicitud falló (${response.status}). Revisa el backend e inténtalo de nuevo.`);
  }
  return response.json();
}

const initial = {caseId: '', status: 'not-run', result: null, analysis: null, inspected: null, operation: null, error: ''};
function reducer(state, event) {
  switch (event.type) {
    case 'CASE': return {...initial, caseId: event.id};
    case 'INSPECT': return {...state, inspected: event.index};
    case 'START': return {...state, status: 'running', operation: event.operation, error: ''};
    case 'SUCCESS':
      if (event.caseId !== state.caseId) return state;
      // Computing a ranking does not apply an intervention or reset an existing one.
      const result = event.operation === 'analysis' && state.result ? state.result : event.data;
      return {...state, result, status: result.selected ? 'intervened' : 'ready', operation: null,
        analysis: event.operation === 'analysis' || event.operation === 'infer' ? event.data : state.analysis};
    case 'ERROR': return {...state, status: state.result ? (state.result.selected ? 'intervened' : 'ready') : 'not-run', operation: null, error: event.message};
    default: return state;
  }
}

export function useWorkspace() {
  const [state, dispatch] = useReducer(reducer, initial);
  const request = useRef(null);
  const revision = useRef(0);
  function selectCase(id) {
    revision.current += 1;
    request.current?.abort();
    request.current = null;
    dispatch({type: 'CASE', id});
  }
  async function run(operation, memory) {
    if (request.current || !state.caseId) return;
    const controller = new AbortController();
    request.current = controller;
    const version = revision.current;
    dispatch({type: 'START', operation});
    try {
      // Analysis computes the original condition and all ten independent interventions
      // together. Selecting a memory afterwards only reads this case's session result.
      const endpoint = operation === 'infer' ? 'analysis' : operation;
      const data = await api(`/cases/${encodeURIComponent(state.caseId)}/${endpoint}`, memory ? {memory} : {}, controller.signal);
      if (data.case_id !== state.caseId) throw new Error('El resultado recibido no corresponde al caso seleccionado.');
      if (operation === 'infer' && (data.memories?.length !== 10 ||
          new Set(data.memories.map(m => m.index)).size !== 10 ||
          data.memories.some(m => m.index < 1 || m.index > 10 || m.memory !== `E${String(m.index).padStart(2, '0')}`))) {
        throw new Error('El análisis recibido no contiene las diez intervenciones E01–E10.');
      }
      if (version === revision.current) dispatch({type: 'SUCCESS', caseId: state.caseId, data, operation});
    } catch (error) {
      if (version === revision.current && error.name !== 'AbortError') dispatch({type: 'ERROR', message: error.message});
    } finally {
      if (request.current === controller) request.current = null;
    }
  }
  return {state, run, selectCase, inspect: index => dispatch({type: 'INSPECT', index})};
}
