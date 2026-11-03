export type AnswerLetter = 'A' | 'B' | 'C' | 'D';
export type MemoryName = `E${string}`;
export interface ProfileCorrelation { mean: number; median: number; min: number; max: number }
export interface DistinctiveRegion {
  memory: MemoryName; rank: number; token: number; specificity_z: number;
  sensitivity: number; excess_over_other_memories: number;
  approx_video_time_seconds: number; frameUrl: string | null;
}
export interface MemoryGrounding { memory: MemoryName; max_specificity_z: number; regions: DistinctiveRegion[] }
export interface Prediction {
  prediction: AnswerLetter; probability: number; probabilities: number[];
  answer: string; correct?: boolean | null;
}
export interface MemoryEffect {
  index: number; memory: MemoryName; rank?: number; delta: number; absolute_delta: number;
  flip: boolean; original_answer_probability_after: number; after: Prediction;
}
export interface ReferenceAnalysis {
  source: 'precomputed_validation'; memories: MemoryEffect[];
  baseline: Pick<Prediction, 'prediction' | 'probability'>; top: number; bottom: number;
}
export interface CaseGrounding {
  case_id: string; source: 'precomputed_visual_analysis'; memories: MemoryGrounding[];
  profile_correlation: ProfileCorrelation; reference_analysis: ReferenceAnalysis;
  files: {selectivity: string; summary: string};
}
export interface DemoCase {
  id: string; question_id: string; video_id: string; category: string; question: string;
  options: string[]; correct_option: AnswerLetter;
  reference: {source: 'precomputed_validation'; top_memory: MemoryName; top_abs_effect: number;
    top_flip: boolean; prediction: AnswerLetter; after: AnswerLetter};
}
