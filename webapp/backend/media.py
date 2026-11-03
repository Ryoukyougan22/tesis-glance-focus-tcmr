"""HTTP byte ranges for seeking MP4s with the existing Starlette version."""
import re
from pathlib import Path
from fastapi.responses import FileResponse, Response, StreamingResponse


def video_response(path, request):
    path = Path(path)
    size = path.stat().st_size
    headers = {'Accept-Ranges': 'bytes'}
    requested = request.headers.get('range')
    if requested is None:
        return FileResponse(path, media_type='video/mp4', headers=headers, method=request.method)
    match = re.fullmatch(r'bytes=(\d*)-(\d*)', requested.strip())
    try:
        if not match or not any(match.groups()):
            raise ValueError('Invalid range')
        first, last = match.groups()
        if first:
            start = int(first)
            end = min(int(last), size - 1) if last else size - 1
        else:
            length = int(last)
            if length <= 0:
                raise ValueError('Invalid suffix')
            start, end = max(0, size - length), size - 1
        if start > end or start >= size:
            raise ValueError('Unsatisfiable range')
    except ValueError:
        return Response(status_code=416, headers={**headers, 'Content-Range': 'bytes */{}'.format(size)})
    headers.update({'Content-Range': 'bytes {}-{}/{}'.format(start, end, size), 'Content-Length': str(end-start+1)})
    if request.method == 'HEAD':
        return Response(status_code=206, media_type='video/mp4', headers=headers)
    def chunks():
        with path.open('rb') as stream:
            stream.seek(start)
            remaining = end-start+1
            while remaining:
                chunk = stream.read(min(65536, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk
    return StreamingResponse(chunks(), status_code=206, media_type='video/mp4', headers=headers)
