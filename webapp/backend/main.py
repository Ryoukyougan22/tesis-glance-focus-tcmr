from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .service import ModelService
from .visual_analysis import VisualAnalysisStore
from .media import video_response


@asynccontextmanager
async def lifespan(app):
    app.state.visual = VisualAnalysisStore()
    app.state.model = ModelService(records=app.state.visual.runtime_cases())
    yield
    del app.state.model
    del app.state.visual


app = FastAPI(title='VideoQA · Glance-Focus + TCMR', lifespan=lifespan)


class Intervention(BaseModel):
    memory: int = Field(..., ge=1, le=10)


def service(case_id=None):
    model = app.state.model
    if case_id is not None and case_id not in model.cases:
        raise HTTPException(404, 'Caso STAR no encontrado')
    return model


@app.get('/api/health')
def health():
    return {'status': 'ready', 'device': str(service().args.device), 'source': 'live_model'}


@app.get('/api/cases')
def cases():
    return [service().public_case(c) for c in service().cases.values()]


@app.api_route('/api/cases/{case_id}/video', methods=['GET', 'HEAD'])
def video(case_id: str, request: Request):
    return video_response(service(case_id).cases[case_id]['video'], request)


@app.post('/api/cases/{case_id}/infer')
def infer(case_id: str):
    return service(case_id).evaluate(case_id)


@app.get('/api/cases/{case_id}/thumbnail')
def thumbnail(case_id: str):
    service(case_id)
    path = app.state.visual.thumbnail(case_id)
    if not path:
        raise HTTPException(404, 'Miniatura no disponible')
    if not path.is_file():
        raise HTTPException(404, 'Miniatura no disponible')
    return FileResponse(path, media_type='image/jpeg')


@app.get('/api/cases/{case_id}/grounding')
def grounding(case_id: str):
    service(case_id)
    data = app.state.visual.groundings.get(case_id)
    if data is None:
        raise HTTPException(503, 'Análisis visual precomputado no disponible para este caso')
    return data


@app.get('/api/cases/{case_id}/grounding/{memory}/frames/{rank}')
def grounding_frame(case_id: str, memory: str, rank: int):
    service(case_id)
    path = app.state.visual.frames.get((case_id, memory, rank))
    if path is None or not path.is_file():
        raise HTTPException(404, 'Frame precomputado no disponible')
    return FileResponse(path, media_type='image/jpeg')


@app.post('/api/cases/{case_id}/analysis')
def analysis(case_id: str):
    return service(case_id).evaluate(case_id, ranking=True)


@app.post('/api/cases/{case_id}/intervene')
def intervene(case_id: str, body: Intervention):
    return service(case_id).evaluate(case_id, selected=body.memory)


@app.post('/api/cases/{case_id}/restore')
def restore(case_id: str):
    return service(case_id).evaluate(case_id)
