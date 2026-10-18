// 评测度量：关键词口径与检索指标计算。独立成模块以便单测（量尺本身需要校准）。
import {
  fuseChannels,
  type ChannelFetches,
  type ScoringParams,
  type SearchResultItem,
} from "../src/index";

export const INTENTS = ["exact", "paraphrase", "temporal", "entity-bridge", "abstention"] as const;
export type Intent = (typeof INTENTS)[number];

export interface EvalQuery {
  id: string;
  intent: Intent;
  q: string;
  expect: string[];
}

/** covered：fetch 时核对语料库中是否存在含 expect 关键词的记忆；未覆盖说明抽取缺失，不计入检索指标 */
export interface CacheEntry extends EvalQuery {
  covered: boolean;
  fetches: ChannelFetches;
}

export interface Metrics {
  hit1: number;
  hit3: number;
  mrr: number;
  /** abstention 查询中产生任意召回的比例 */
  fp: number;
}

/** 关键词匹配口径：小写 + 去空白后做子串包含 */
export function norm(s: string): string {
  return s.toLowerCase().replace(/\s+/g, "");
}

export function containsAny(text: string, keywords: string[]): boolean {
  const t = norm(text);
  return keywords.some((k) => t.includes(norm(k)));
}

/** 实体对匹配：双向子串（"鸡肉冻干"↔"冻干"、"汉源花椒"↔"花椒" 均视为命中） */
export function matchesTerm(entity: string, expected: string): boolean {
  const a = norm(entity);
  const b = norm(expected);
  return a.includes(b) || b.includes(a);
}

export function judge(
  entry: CacheEntry,
  results: SearchResultItem[],
): { hit1: number; hit3: number; rr: number; fp: number } {
  if (entry.intent === "abstention") {
    return { hit1: 0, hit3: 0, rr: 0, fp: results.length > 0 ? 1 : 0 };
  }
  const hitN = (n: number) =>
    results.slice(0, n).some((r) => containsAny(r.memory, entry.expect)) ? 1 : 0;
  let rr = 0;
  for (let i = 0; i < results.length; i++) {
    if (containsAny(results[i].memory, entry.expect)) {
      rr = 1 / (i + 1);
      break;
    }
  }
  return { hit1: hitN(1), hit3: hitN(3), rr, fp: 0 };
}

export function evaluate(
  entries: CacheEntry[],
  overrides: Partial<ScoringParams>,
  topK: number,
): Metrics {
  let hit1 = 0,
    hit3 = 0,
    mrr = 0,
    fp = 0,
    n = 0,
    nAbst = 0;
  for (const e of entries) {
    const results = fuseChannels(e.fetches, overrides, topK);
    const j = judge(e, results);
    if (e.intent === "abstention") {
      fp += j.fp;
      nAbst++;
    } else {
      hit1 += j.hit1;
      hit3 += j.hit3;
      mrr += j.rr;
      n++;
    }
  }
  return {
    hit1: n ? hit1 / n : 0,
    hit3: n ? hit3 / n : 0,
    mrr: n ? mrr / n : 0,
    fp: nAbst ? fp / nAbst : 0,
  };
}
