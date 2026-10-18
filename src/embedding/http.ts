import type { EmbeddingResult } from "../types";

export interface StandardEmbeddingConfig {
  baseURL: string;
  apiKey: string;
  model: string;
  dims: number;
}

/** 兼容两种响应形态：批量时 data 为数组；方舟 multimodal 端点单条输入时 data 为对象 */
export function parseEmbedResponse(json: any, dims: number): EmbeddingResult[] {
  const arr: any[] = Array.isArray(json.data) ? json.data : [json.data];
  const byIndex = [...arr].sort((a, b) => (a.index ?? 0) - (b.index ?? 0));
  return byIndex.map((d) => {
    if (!Array.isArray(d.embedding) || d.embedding.length !== dims) {
      throw new Error(`embedding 维度不符：期望 ${dims}，实际 ${d.embedding?.length}`);
    }
    const list: Array<{ index: number; value: number }> | undefined = d.sparse_embedding;
    return {
      dense: d.embedding as number[],
      ...(Array.isArray(list) && list.length > 0
        ? {
            sparse: {
              indices: list.map((e) => e.index),
              values: list.map((e) => e.value),
            },
          }
        : {}),
    };
  });
}

/** 瞬时故障（限流/网关类）应退避重试，配置类错误（4xx）不重试 */
const RETRYABLE_STATUSES = new Set([429, 500, 502, 503, 504]);
const MAX_RETRY_DELAY_MS = 5000;

/** 重试耗尽后返回最后一个失败响应，由调用方按自身语义处理（报错或降级） */
export async function fetchWithBackoff(
  url: string,
  init: RequestInit,
  attempts = 4,
): Promise<Response> {
  let last: Response | undefined;
  for (let i = 0; i < attempts; i++) {
    const res = await fetch(url, init);
    if (res.ok || !RETRYABLE_STATUSES.has(res.status)) return res;
    last = res;
    await res.text().catch(() => "");
    if (i < attempts - 1) {
      const retryAfter = Number(res.headers.get("retry-after"));
      const delay =
        Number.isFinite(retryAfter) && retryAfter > 0
          ? Math.min(retryAfter * 1000, MAX_RETRY_DELAY_MS)
          : Math.min(500 * 2 ** i + Math.random() * 250, MAX_RETRY_DELAY_MS);
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  }
  return last!;
}

/** 标准 /embeddings：input 数组为真批量语义 */
export async function embedDenseBatch(
  config: StandardEmbeddingConfig,
  texts: string[],
): Promise<EmbeddingResult[]> {
  const res = await fetchWithBackoff(`${config.baseURL}/embeddings`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${config.apiKey}`,
    },
    body: JSON.stringify({
      model: config.model,
      input: texts,
      encoding_format: "float",
    }),
  });
  if (!res.ok) {
    throw new Error(`embedding 请求失败: HTTP ${res.status} ${await res.text().catch(() => "")}`);
  }
  return parseEmbedResponse(await res.json(), config.dims);
}
