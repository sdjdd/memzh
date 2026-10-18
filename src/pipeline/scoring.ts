// 纯打分层：不依赖网络与存储。
// 评测的参数网格对本模块离线重放：取数一次，参数扫描无 API 开销。
import type { VectorHit, VectorPayload } from "../types";

export interface ScoringParams {
  /** 语义分门控：低于阈值的候选在融合前直接淘汰，其余通道无法弥补 */
  threshold: number;
  /** BM25 sigmoid 中点/斜率，显式给值时跳过按词数的自适应分档 */
  bm25Midpoint?: number;
  bm25Steepness?: number;
  entityMinSim: number;
  entityBoostWeight: number;
}

/**
 * 默认打分参数，经 eval/ 评测集校准（33 条语料 / 31 查询，2026-10-18）：
 * - threshold 0.4：实测无关 query 的语义噪声地板为 0.270~0.386，正确候选 min 0.392、
 *   p25 0.501；0.3 下弃答查询误报率 0.8，证伪
 * - entityMinSim 0.5：网格中 0.5/0.7 指标无差异，暂维持原值
 * - BM25 中点未显式给值时按 query 词数自适应分档（见 getBm25Params）
 */
export const DEFAULT_SCORING: ScoringParams = {
  threshold: 0.4,
  entityMinSim: 0.5,
  entityBoostWeight: 0.5,
};

/**
 * 按 query 词数选择 sigmoid 参数（长 query 原始分偏高，中点随之上移）。
 * 中点已校准到模型学习权重的点积量级（实测原始分 p50≈0.2、max≈0.7；
 * 原 5.0~12.0 为英文 BM25 词频量级，偏大约 25 倍，bm25 分量恒近似 0.02）。
 * steepness 保持 0.5~0.7 的平坦区：实测陡峭 sigmoid（≥2）会让 bm25 高的
 * 次优候选翻转 top1，hit@1 下降。
 */
export function getBm25Params(queryWordCount: number): [number, number] {
  const n = Math.max(queryWordCount, 1);
  if (n <= 3) return [0.2, 0.7];
  if (n <= 6) return [0.28, 0.6];
  if (n <= 9) return [0.36, 0.5];
  if (n <= 15) return [0.4, 0.5];
  return [0.48, 0.5];
}

export function normalizeBm25(rawScore: number, midpoint: number, steepness: number): number {
  return 1.0 / (1.0 + Math.exp(-steepness * (rawScore - midpoint)));
}

/** 实体通道的原始命中：某实体（与 query 候选的相似度 sim）关联着某条记忆 */
export interface EntityMatch {
  memoryId: string;
  sim: number;
  /** 该实体关联的记忆总数（热门度衰减用） */
  linkedCount: number;
}

/**
 * 实体 boost = sim × weight × 衰减因子 1/(1+0.001(n-1)²)，n 为该实体关联的记忆数。
 * 衰减因子用于抑制关联记忆过多的通用实体；同一记忆被多个实体命中时取最大值（不叠加）。
 */
export function computeEntityBoosts(
  matches: EntityMatch[],
  params: ScoringParams,
): Record<string, number> {
  const boosts: Record<string, number> = {};
  for (const m of matches) {
    if (m.sim < params.entityMinSim) continue;
    const n = Math.max(m.linkedCount, 1);
    const weight = 1.0 / (1.0 + 0.001 * (n - 1) ** 2);
    const boost = m.sim * params.entityBoostWeight * weight;
    boosts[m.memoryId] = Math.max(boosts[m.memoryId] ?? 0, boost);
  }
  return boosts;
}

export interface ScoreDetails {
  semanticScore: number;
  bm25Score: number;
  entityBoost: number;
  rawScore: number;
  maxPossibleScore: number;
  finalScore: number;
  threshold: number;
}

export interface ScoredResult {
  id: string;
  score: number;
  payload: VectorPayload;
  scoreDetails?: ScoreDetails;
}

/**
 * 加性融合：final = (sem + bm25 + ent) / max_possible。
 * 分母自适应：语义恒 1.0；有 BM25 命中 +1.0；有实体 boost +0.5。
 */
export function scoreAndRank(
  semanticResults: VectorHit[],
  bm25Scores: Record<string, number>,
  entityBoosts: Record<string, number>,
  params: ScoringParams,
  topK: number,
  explain = false,
): ScoredResult[] {
  const hasBm25 = Object.keys(bm25Scores).length > 0;
  const hasEntity = Object.keys(entityBoosts).length > 0;

  let maxPossible = 1.0;
  if (hasBm25) maxPossible += 1.0;
  if (hasEntity) maxPossible += params.entityBoostWeight;

  const scored: ScoredResult[] = [];
  for (const result of semanticResults) {
    const semanticScore = result.score ?? 0;
    if (semanticScore < params.threshold) continue;

    const bm25Score = bm25Scores[result.id] ?? 0;
    const entityBoost = entityBoosts[result.id] ?? 0;

    const rawCombined = semanticScore + bm25Score + entityBoost;
    const combined = Math.min(rawCombined / maxPossible, 1.0);

    scored.push({
      id: result.id,
      score: combined,
      payload: result.payload,
      ...(explain && {
        scoreDetails: {
          semanticScore,
          bm25Score,
          entityBoost,
          rawScore: rawCombined,
          maxPossibleScore: maxPossible,
          finalScore: combined,
          threshold: params.threshold,
        },
      }),
    });
  }
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, topK);
}
