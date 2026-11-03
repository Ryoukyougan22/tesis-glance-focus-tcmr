import React, { useState, type RefObject } from 'react';
import type { DistinctiveRegion, MemoryGrounding } from './types';
import VisualTimeline from './VisualTimeline';

interface Props {caseId: string; videoRef: RefObject<HTMLVideoElement | null>; memory: MemoryGrounding | undefined;
  activeToken: number | null; onSeek: (r: DistinctiveRegion) => void; onError: () => void}
export default function GroundedVideo({caseId, videoRef, memory, activeToken, onSeek, onError}: Props) {
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  return <div className="grounded-video"><video ref={videoRef} controls preload="metadata" onError={onError}
    src={`/api/cases/${encodeURIComponent(caseId)}/video`}
    onLoadedMetadata={event => setDuration(Number.isFinite(event.currentTarget.duration) ? event.currentTarget.duration : 0)}
    onTimeUpdate={event => setCurrentTime(event.currentTarget.currentTime)}/>
    <VisualTimeline duration={duration} currentTime={currentTime} memory={memory} activeToken={activeToken} onSeek={onSeek}
      onScrub={time => {if (videoRef.current) videoRef.current.currentTime = time;}}/>
  </div>;
}
