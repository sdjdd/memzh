import { afterEach, describe, expect, it, vi } from "vitest";
import { fetchWithBackoff, parseEmbedResponse } from "./http";

describe("parseEmbedResponse", () => {
  const dims = 4;
  const emb = (offset: number) => Array.from({ length: dims }, (_, i) => offset + i);

  it("批量形态：data 为数组，按 index 排序", () => {
    const out = parseEmbedResponse(
      {
        data: [
          { index: 1, embedding: emb(10) },
          { index: 0, embedding: emb(0) },
        ],
      },
      dims,
    );
    expect(out).toHaveLength(2);
    expect(out[0].dense).toEqual(emb(0));
    expect(out[1].dense).toEqual(emb(10));
  });

  it("单条形态：data 为对象（方舟 multimodal）", () => {
    const out = parseEmbedResponse({ data: { index: 0, embedding: emb(0) } }, dims);
    expect(out).toHaveLength(1);
    expect(out[0].dense).toEqual(emb(0));
  });

  it("sparse_embedding 透传为 indices/values", () => {
    const out = parseEmbedResponse(
      {
        data: {
          index: 0,
          embedding: emb(0),
          sparse_embedding: [
            { index: 7, value: 0.5 },
            { index: 3, value: 0.9 },
          ],
        },
      },
      dims,
    );
    expect(out[0].sparse).toEqual({ indices: [7, 3], values: [0.5, 0.9] });
  });

  it("无 sparse 或空 sparse 时结果不含 sparse 字段", () => {
    const noSparse = parseEmbedResponse({ data: { index: 0, embedding: emb(0) } }, dims);
    expect(noSparse[0].sparse).toBeUndefined();
    const emptySparse = parseEmbedResponse(
      {
        data: { index: 0, embedding: emb(0), sparse_embedding: [] },
      },
      dims,
    );
    expect(emptySparse[0].sparse).toBeUndefined();
  });

  it("维度不符立即抛错", () => {
    expect(() => parseEmbedResponse({ data: { index: 0, embedding: [1, 2, 3] } }, dims)).toThrow(
      /维度不符/,
    );
  });
});

describe("fetchWithBackoff", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("非重试类状态码立即返回，不重试", async () => {
    const fetchMock = vi.fn(async () => new Response("nope", { status: 400 }));
    vi.stubGlobal("fetch", fetchMock);
    const res = await fetchWithBackoff("http://unit-test", { method: "POST" }, 3);
    expect(res.status).toBe(400);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("重试耗尽后返回最后一个失败响应（不抛错）", async () => {
    const fetchMock = vi.fn(async () => new Response("rate limited", { status: 429 }));
    vi.stubGlobal("fetch", fetchMock);
    const res = await fetchWithBackoff("http://unit-test", { method: "POST" }, 2);
    expect(res.status).toBe(429);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("瞬时失败后重试成功则返回成功响应", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(new Response("rate limited", { status: 429 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ ok: true }), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    const res = await fetchWithBackoff("http://unit-test", { method: "POST" }, 2);
    expect(res.ok).toBe(true);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});
