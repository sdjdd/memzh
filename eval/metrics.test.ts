import { describe, expect, it } from "vitest";
import { containsAny, evaluate, judge, matchesTerm, norm, type CacheEntry } from "./metrics";
import type { VectorHit } from "../src/types";

const semHit = (id: string, score: number, data: string): VectorHit => ({
  id,
  score,
  payload: { data },
});

const entry = (partial: Partial<CacheEntry>): CacheEntry => ({
  id: "q1",
  intent: "exact",
  q: "query",
  expect: ["青椒"],
  covered: true,
  fetches: {
    query: "query",
    semantic: [semHit("m1", 0.9, "用户喜欢吃青椒")],
    bm25: null,
    entityMatches: [],
    queryTokenCount: 2,
  },
  ...partial,
});

describe("关键词口径", () => {
  it("norm：小写 + 去空白", () => {
    expect(norm("User Likes 青椒 ")).toBe("userlikes青椒");
  });

  it("containsAny：子串包含", () => {
    expect(containsAny("用户喜欢吃青椒", ["青椒"])).toBe(true);
    expect(containsAny("用户喜欢吃茄子", ["青椒"])).toBe(false);
  });

  it("matchesTerm：双向子串", () => {
    expect(matchesTerm("汉源花椒", "花椒")).toBe(true);
    expect(matchesTerm("花椒", "汉源花椒")).toBe(true);
    expect(matchesTerm("青椒", "花椒")).toBe(false);
  });
});

describe("judge", () => {
  it("hit@1 / hit@3 / MRR 按首次命中位次", () => {
    const e = entry({});
    const first = judge(e, [
      { id: "m1", score: 0.9, memory: "用户喜欢吃青椒" },
      { id: "m2", score: 0.5, memory: "无关" },
    ]);
    expect(first).toEqual({ hit1: 1, hit3: 1, rr: 1, fp: 0 });

    const third = judge(e, [
      { id: "x1", score: 0.9, memory: "无关一" },
      { id: "x2", score: 0.8, memory: "无关二" },
      { id: "x3", score: 0.7, memory: "用户喜欢吃青椒" },
    ]);
    expect(third.hit1).toBe(0);
    expect(third.hit3).toBe(1);
    expect(third.rr).toBeCloseTo(1 / 3);

    const miss = judge(e, [
      { id: "x1", score: 0.9, memory: "无关一" },
      { id: "x2", score: 0.8, memory: "无关二" },
    ]);
    expect(miss).toEqual({ hit1: 0, hit3: 0, rr: 0, fp: 0 });
  });

  it("第 4 位命中不计入 hit@3，但计入 MRR", () => {
    const j = judge(entry({}), [
      { id: "a", score: 0.9, memory: "一" },
      { id: "b", score: 0.8, memory: "二" },
      { id: "c", score: 0.7, memory: "三" },
      { id: "d", score: 0.6, memory: "用户喜欢吃青椒" },
    ]);
    expect(j.hit3).toBe(0);
    expect(j.rr).toBeCloseTo(0.25);
  });

  it("abstention：任何召回都计 fp", () => {
    const e = entry({ intent: "abstention", expect: [] });
    expect(judge(e, [{ id: "m1", score: 0.9, memory: "用户喜欢吃青椒" }]).fp).toBe(1);
    expect(judge(e, []).fp).toBe(0);
  });
});

describe("evaluate", () => {
  it("检索指标只平均非弃答查询，fp 只平均弃答查询", () => {
    const hitEntry = entry({ id: "q1" });
    const missEntry = entry({
      id: "q2",
      fetches: {
        query: "query",
        semantic: [semHit("m1", 0.9, "无关记忆")],
        bm25: null,
        entityMatches: [],
        queryTokenCount: 2,
      },
    });
    const abstEntry = entry({
      id: "q3",
      intent: "abstention",
      expect: [],
      fetches: {
        query: "query",
        semantic: [],
        bm25: null,
        entityMatches: [],
        queryTokenCount: 2,
      },
    });
    const m = evaluate([hitEntry, missEntry, abstEntry], {}, 5);
    expect(m.hit1).toBeCloseTo(0.5);
    expect(m.hit3).toBeCloseTo(0.5);
    expect(m.mrr).toBeCloseTo(0.5);
    expect(m.fp).toBeCloseTo(0);
  });

  it("弃答查询产生召回时 fp=1", () => {
    const abstHit = entry({
      id: "q3",
      intent: "abstention",
      expect: [],
      fetches: {
        query: "query",
        semantic: [semHit("m1", 0.9, "用户喜欢吃青椒")],
        bm25: null,
        entityMatches: [],
        queryTokenCount: 2,
      },
    });
    expect(evaluate([abstHit], {}, 5).fp).toBe(1);
  });
});
