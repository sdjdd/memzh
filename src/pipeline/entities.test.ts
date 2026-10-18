import { describe, expect, it, vi } from "vitest";
import { MERGE_SIMILARITY, upsertEntities } from "./entities";
import type { EmbeddingProvider } from "../embedding";
import type { VectorStore } from "../store";
import type { VectorHit, VectorPoint } from "../types";

function fakeStore(hits: VectorHit[]) {
  const calls = {
    setPayload: [] as Array<{ id: string; patch: any }>,
    upsert: [] as VectorPoint[],
  };
  const store: VectorStore = {
    ensureCollection: async () => {},
    upsert: async (points) => {
      calls.upsert.push(...points);
    },
    delete: async () => {},
    getPoints: async () => [],
    searchDense: async () => hits,
    searchSparse: async () => null,
    setPayload: async (id, patch) => {
      calls.setPayload.push({ id, patch });
    },
  };
  return { store, calls };
}

const fakeEmbedder = (): EmbeddingProvider => ({
  name: "fake",
  dims: 4,
  supportsSparse: false,
  embed: async (texts) => texts.map((t) => ({ dense: [t.length, 1, 0, 0] })),
});

const run = async (hits: VectorHit[], items: Array<{ text: string; memoryId: string }>) => {
  const { store, calls } = fakeStore(hits);
  const assignments = await upsertEntities({
    items,
    userId: "u1",
    store,
    embedder: fakeEmbedder(),
  });
  return { assignments, calls };
};

describe("upsertEntities", () => {
  it("归一化精确匹配合并：setPayload 追加关联，不新建", async () => {
    const { assignments, calls } = await run(
      [
        {
          id: "ent-1",
          score: 0.99,
          payload: { data: "青椒", linkedMemoryIds: ["mem-old"] },
        },
      ],
      [{ text: " 青椒 ", memoryId: "mem-new" }],
    );
    expect(calls.upsert).toHaveLength(0);
    expect(calls.setPayload[0]).toEqual({
      id: "ent-1",
      patch: { linkedMemoryIds: ["mem-old", "mem-new"] },
    });
    expect(assignments[0]).toEqual({
      text: "青椒",
      entityId: "ent-1",
      memoryIds: ["mem-old", "mem-new"],
    });
  });

  it("无精确匹配但语义达标（≥ MERGE_SIMILARITY）时合并 top1", async () => {
    const { assignments, calls } = await run(
      [
        {
          id: "ent-2",
          score: MERGE_SIMILARITY,
          payload: { data: "鸡肉冻干", linkedMemoryIds: [] },
        },
      ],
      [{ text: "冻干鸡肉", memoryId: "mem-1" }],
    );
    expect(calls.upsert).toHaveLength(0);
    expect(assignments[0].entityId).toBe("ent-2");
  });

  it("相似度低于阈值时新建实体点（payload 归一化 + md5 hash）", async () => {
    const { assignments, calls } = await run(
      [
        {
          id: "ent-3",
          score: MERGE_SIMILARITY - 0.01,
          payload: { data: "别的实体" },
        },
      ],
      [{ text: "花椒", memoryId: "mem-1" }],
    );
    expect(calls.setPayload).toHaveLength(0);
    expect(calls.upsert).toHaveLength(1);
    const point = calls.upsert[0];
    expect(point.payload.data).toBe("花椒");
    expect(point.payload.linkedMemoryIds).toEqual(["mem-1"]);
    expect(point.payload.userId).toBe("u1");
    expect(point.payload.hash).toMatch(/^[0-9a-f]{32}$/);
    expect(assignments[0].entityId).toBe(point.id);
  });

  it("精确匹配优先于排序位次（top1 相似度低也让位）", async () => {
    const { assignments } = await run(
      [
        { id: "low", score: 0.5, payload: { data: "其他" } },
        { id: "exact-hit", score: 0.6, payload: { data: "花椒" } },
      ],
      [{ text: "花椒", memoryId: "mem-1" }],
    );
    expect(assignments[0].entityId).toBe("exact-hit");
  });

  it("批内同文本聚合：只检索一次，关联全部 memoryId", async () => {
    const { store, calls } = fakeStore([]);
    const searchDense = vi.spyOn(store, "searchDense");
    const assignments = await upsertEntities({
      items: [
        { text: "花椒", memoryId: "mem-1" },
        { text: " 花椒 ", memoryId: "mem-2" },
      ],
      userId: "u1",
      store,
      embedder: fakeEmbedder(),
    });
    expect(searchDense).toHaveBeenCalledTimes(1);
    expect(calls.upsert[0].payload.linkedMemoryIds).toEqual(["mem-1", "mem-2"]);
    expect(assignments[0].memoryIds).toEqual(["mem-1", "mem-2"]);
  });

  it("单实体失败仅告警，不阻塞其余实体", async () => {
    const broken: EmbeddingProvider = {
      ...fakeEmbedder(),
      embed: async (texts) => {
        if (texts[0] === "青椒") throw new Error("embed fail");
        return texts.map((t) => ({ dense: [t.length, 1, 0, 0] }));
      },
    };
    const { store } = fakeStore([]);
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const assignments = await upsertEntities({
      items: [
        { text: "青椒", memoryId: "mem-1" },
        { text: "花椒", memoryId: "mem-2" },
      ],
      userId: "u1",
      store,
      embedder: broken,
    });
    warn.mockRestore();
    expect(assignments.map((a) => a.text)).toEqual(["花椒"]);
  });
});
