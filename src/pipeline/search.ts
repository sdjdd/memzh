// 检索管线，刻意拆分为两部分：
//   fetchChannels —— 三通道原始取数，结果可整体缓存，供评测离线重放；
//   fuseChannels  —— 纯打分，评测的参数网格仅作用于本函数，无 API 开销。
import type { EmbeddingProvider } from "../embedding";
import type { VectorStore } from "../store";
import type { VectorHit } from "../types";
import { tokenizeForCandidates } from "../tokenize";
import {
  computeEntityBoosts,
  DEFAULT_SCORING,
  getBm25Params,
  normalizeBm25,
  scoreAndRank,
  type EntityMatch,
  type ScoredResult,
  type ScoringParams,
} from "./scoring";

export interface SearchOptions {
  userId: string;
  topK?: number;
  explain?: boolean;
  scoring?: Partial<ScoringParams>;
}

export interface SearchResultItem {
  id: string;
  memory: string;
  score: number;
  createdAt?: string;
  attributedTo?: "user" | "assistant";
  scoreDetails?: ScoredResult["scoreDetails"];
}

export interface SearchDeps {
  store: VectorStore;
  entityStore: VectorStore;
  embedder: EmbeddingProvider;
}

/** 整体可 JSON 序列化（离线重放的前提） */
export interface ChannelFetches {
  query: string;
  /** jieba 词数（BM25 sigmoid 自适应分档用） */
  queryTokenCount: number;
  semantic: VectorHit[];
  /** null 表示稀疏通道不可用（provider 未提供稀疏、存储不支持或检索失败） */
  bm25: VectorHit[] | null;
  /** 实体通道原始命中，未过门槛 */
  entityMatches: EntityMatch[];
}

export async function fetchChannels(
  query: string,
  deps: SearchDeps,
  opts: SearchOptions,
): Promise<ChannelFetches> {
  const topK = opts.topK ?? 20;

  // 原句直接嵌入，不做预处理
  const [queryVec] = await deps.embedder.embed([query]);

  // 候选池 over-fetch（公式沿用 mem0）
  const internalLimit = Math.max(topK * 4, 60);
  const semantic = await deps.store.searchDense(queryVec.dense, internalLimit, {
    userId: opts.userId,
  });

  const bm25 = queryVec.sparse
    ? await deps.store.searchSparse(queryVec.sparse, internalLimit, {
        userId: opts.userId,
      })
    : null;

  // 实体通道：分词候选（当前简化为长度 ≥2 的 token，去重后封顶 8 个）
  const entityMatches: EntityMatch[] = [];
  try {
    const candidates = [
      ...new Set(tokenizeForCandidates(query).filter((t) => t.length >= 2)),
    ].slice(0, 8);
    if (candidates.length > 0) {
      const vecs = await deps.embedder.embed(candidates);
      for (let i = 0; i < candidates.length; i++) {
        // 先尽量召回相似实体，再由门槛过滤
        const hits = await deps.entityStore.searchDense(vecs[i].dense, 100, {
          userId: opts.userId,
        });
        for (const hit of hits) {
          const linked = hit.payload.linkedMemoryIds ?? [];
          for (const memoryId of linked) {
            entityMatches.push({
              memoryId,
              sim: hit.score ?? 0,
              linkedCount: linked.length,
            });
          }
        }
      }
    }
  } catch (e) {
    console.warn("实体通道失败，继续语义+BM25:", e);
  }

  return {
    query,
    queryTokenCount: tokenizeForCandidates(query).length,
    semantic,
    bm25,
    entityMatches,
  };
}

export function fuseChannels(
  raw: ChannelFetches,
  overrides?: Partial<ScoringParams>,
  topK = 20,
  explain = false,
): SearchResultItem[] {
  const params: ScoringParams = { ...DEFAULT_SCORING, ...overrides };

  // BM25 分量：sigmoid 归一，同一记忆多次命中取最大
  const bm25Scores: Record<string, number> = {};
  if (raw.bm25) {
    const [midpoint, steepness] =
      params.bm25Midpoint !== undefined && params.bm25Steepness !== undefined
        ? [params.bm25Midpoint, params.bm25Steepness]
        : getBm25Params(raw.queryTokenCount);
    for (const mem of raw.bm25) {
      const score = mem.score ?? 0;
      if (score > 0) {
        bm25Scores[mem.id] = Math.max(
          bm25Scores[mem.id] ?? 0,
          normalizeBm25(score, midpoint, steepness),
        );
      }
    }
  }

  const entityBoosts = computeEntityBoosts(raw.entityMatches, params);

  const scored = scoreAndRank(raw.semantic, bm25Scores, entityBoosts, params, topK, explain);

  return scored
    .filter((s) => s.payload?.data)
    .map((s) => ({
      id: s.id,
      memory: s.payload.data,
      score: s.score,
      createdAt: s.payload.createdAt,
      attributedTo: s.payload.attributedTo,
      ...(s.scoreDetails ? { scoreDetails: s.scoreDetails } : {}),
    }));
}

export async function searchMemories(
  query: string,
  deps: SearchDeps,
  opts: SearchOptions,
): Promise<SearchResultItem[]> {
  const raw = await fetchChannels(query, deps, opts);
  return fuseChannels(raw, opts.scoring, opts.topK ?? 20, opts.explain ?? false);
}
