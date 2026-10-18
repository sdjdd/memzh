// /embeddings/multimodal 的 input 数组是融合编码语义而非批量，一次只能编码一条：
// 多条文本会被融合为单个向量，语义错误且不报错。
import type { EmbeddingResult } from "../types";
import type { EmbeddingProvider, EmbeddingProviderConfig } from "./index";
import {
  embedDenseBatch,
  fetchWithBackoff,
  parseEmbedResponse,
  type StandardEmbeddingConfig,
} from "./http";
import { mapWithConcurrency, warnOnce } from "../utils";

export type VolcengineEmbedderConfig = Omit<EmbeddingProviderConfig, "provider">;

export function createVolcengineEmbedder(config: VolcengineEmbedderConfig): EmbeddingProvider {
  const base: StandardEmbeddingConfig = {
    baseURL: config.baseURL,
    apiKey: config.apiKey,
    model: config.model,
    dims: config.dims,
  };
  const maxBatchSize = config.maxBatchSize ?? 32;
  const concurrency = config.concurrency ?? 4;
  const wantSparse = config.sparse ?? false;
  // multimodal 失败后置为不可用，本实例内不再重试，避免逐条发出注定失败的请求
  let multimodalUsable = wantSparse;

  async function embedOneWithSparse(text: string) {
    if (multimodalUsable) {
      try {
        const res = await fetchWithBackoff(`${base.baseURL}/embeddings/multimodal`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${base.apiKey}`,
          },
          body: JSON.stringify({
            model: base.model,
            input: [{ type: "text", text }],
            sparse_embedding: { type: "enabled" },
            encoding_format: "float",
          }),
        });
        if (res.ok) {
          const [r] = parseEmbedResponse(await res.json(), base.dims);
          if (!r?.sparse) {
            warnOnce(
              "volcengine:no-sparse-field",
              "volcengine provider 未返回 sparse_embedding（模型版本或开关不支持），退化为纯稠密向量",
            );
          }
          return r;
        }
        warnOnce(
          "volcengine:multimodal-rejected",
          `volcengine 稀疏向量请求失败（HTTP ${res.status}），降级为标准 /embeddings 纯稠密模式`,
        );
        multimodalUsable = false;
      } catch (e) {
        warnOnce("volcengine:multimodal-error", `volcengine multimodal 端点异常，降级纯稠密: ${e}`);
        multimodalUsable = false;
      }
    }
    const [r] = await embedDenseBatch(base, [text]);
    return r;
  }

  return {
    name: "volcengine",
    dims: config.dims,
    supportsSparse: wantSparse,
    async embed(texts) {
      if (texts.length === 0) return [];
      if (multimodalUsable) {
        return mapWithConcurrency(texts, concurrency, embedOneWithSparse);
      }
      const out: EmbeddingResult[] = [];
      for (let i = 0; i < texts.length; i += maxBatchSize) {
        out.push(...(await embedDenseBatch(base, texts.slice(i, i + maxBatchSize))));
      }
      return out;
    },
  };
}
