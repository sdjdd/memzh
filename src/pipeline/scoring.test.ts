import { describe, expect, it } from "vitest";
import {
  computeEntityBoosts,
  DEFAULT_SCORING,
  getBm25Params,
  normalizeBm25,
  scoreAndRank,
  type ScoringParams,
} from "./scoring";
import type { VectorHit } from "../types";

const hit = (id: string, score: number): VectorHit => ({
  id,
  score,
  payload: { data: `记忆${id}` },
});

describe("getBm25Params 分档", () => {
  it("按词数落桶（含边界）", () => {
    expect(getBm25Params(1)).toEqual([0.2, 0.7]);
    expect(getBm25Params(3)).toEqual([0.2, 0.7]);
    expect(getBm25Params(4)).toEqual([0.28, 0.6]);
    expect(getBm25Params(6)).toEqual([0.28, 0.6]);
    expect(getBm25Params(7)).toEqual([0.36, 0.5]);
    expect(getBm25Params(9)).toEqual([0.36, 0.5]);
    expect(getBm25Params(10)).toEqual([0.4, 0.5]);
    expect(getBm25Params(15)).toEqual([0.4, 0.5]);
    expect(getBm25Params(16)).toEqual([0.48, 0.5]);
  });

  it("0 与负数按 1 处理", () => {
    expect(getBm25Params(0)).toEqual([0.2, 0.7]);
    expect(getBm25Params(-5)).toEqual([0.2, 0.7]);
  });
});

describe("normalizeBm25 sigmoid", () => {
  it("raw 等于中点时为 0.5", () => {
    expect(normalizeBm25(0.2, 0.2, 0.7)).toBeCloseTo(0.5);
  });

  it("严格单调递增，值域开区间 (0,1)", () => {
    let prev = 0;
    for (let raw = -10; raw <= 10; raw += 0.5) {
      const v = normalizeBm25(raw, 0.2, 0.7);
      expect(v).toBeGreaterThan(prev);
      expect(v).toBeGreaterThan(0);
      expect(v).toBeLessThan(1);
      prev = v;
    }
  });
});

describe("computeEntityBoosts", () => {
  const params: ScoringParams = {
    threshold: 0.4,
    entityMinSim: 0.5,
    entityBoostWeight: 0.5,
  };

  it("低于 entityMinSim 的命中被过滤", () => {
    expect(computeEntityBoosts([{ memoryId: "m1", sim: 0.49, linkedCount: 1 }], params)).toEqual(
      {},
    );
  });

  it("boost = sim × weight × 衰减因子（n=1 无衰减）", () => {
    const boosts = computeEntityBoosts([{ memoryId: "m1", sim: 0.8, linkedCount: 1 }], params);
    expect(boosts.m1).toBeCloseTo(0.4);
  });

  it("关联记忆越多衰减越强", () => {
    const small = computeEntityBoosts([{ memoryId: "m1", sim: 0.8, linkedCount: 1 }], params).m1!;
    const large = computeEntityBoosts([{ memoryId: "m1", sim: 0.8, linkedCount: 100 }], params).m1!;
    expect(large).toBeLessThan(small);
  });

  it("同一记忆被多个实体命中取最大（不叠加）", () => {
    const boosts = computeEntityBoosts(
      [
        { memoryId: "m1", sim: 0.6, linkedCount: 1 },
        { memoryId: "m1", sim: 0.9, linkedCount: 1 },
      ],
      params,
    );
    expect(boosts.m1).toBeCloseTo(0.45);
  });
});

describe("scoreAndRank", () => {
  const base: ScoringParams = { ...DEFAULT_SCORING };

  it("语义分低于阈值的候选在融合前淘汰（其余通道无法弥补）", () => {
    const out = scoreAndRank([hit("m1", 0.39)], { m1: 1.0 }, { m1: 0.5 }, base, 5);
    expect(out).toHaveLength(0);
  });

  it("分母自适应：仅语义命中时为 1.0", () => {
    const out = scoreAndRank([hit("m1", 0.8)], {}, {}, base, 5, true);
    expect(out[0].score).toBeCloseTo(0.8);
    expect(out[0].scoreDetails!.maxPossibleScore).toBe(1.0);
  });

  it("语义 + BM25 → 分母 2.0", () => {
    const out = scoreAndRank([hit("m1", 0.8)], { m1: 0.6 }, {}, base, 5, true);
    expect(out[0].score).toBeCloseTo(1.4 / 2.0);
    expect(out[0].scoreDetails!.maxPossibleScore).toBe(2.0);
  });

  it("语义 + BM25 + 实体 → 分母 2.5", () => {
    const out = scoreAndRank([hit("m1", 0.8)], { m1: 0.6 }, { m1: 0.5 }, base, 5, true);
    expect(out[0].score).toBeCloseTo(1.9 / 2.5);
    expect(out[0].scoreDetails!.maxPossibleScore).toBe(2.5);
  });

  it("融合分封顶 1.0", () => {
    const out = scoreAndRank([hit("m1", 0.9)], { m1: 1.0 }, { m1: 0.5 }, base, 5);
    expect(out[0].score).toBeLessThanOrEqual(1.0);
  });

  it("按融合分降序并截断 topK", () => {
    const out = scoreAndRank(
      [hit("a", 0.5), hit("b", 0.9), hit("c", 0.7), hit("d", 0.6)],
      {},
      {},
      base,
      2,
    );
    expect(out.map((r) => r.id)).toEqual(["b", "c"]);
  });

  it("explain=false 不携带 scoreDetails", () => {
    const out = scoreAndRank([hit("m1", 0.8)], {}, {}, base, 5);
    expect(out[0].scoreDetails).toBeUndefined();
  });
});
