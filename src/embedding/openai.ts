import type { EmbeddingResult } from "../types";
import type { EmbeddingProvider, EmbeddingProviderConfig } from "./index";
import { embedDenseBatch, type StandardEmbeddingConfig } from "./http";

export type OpenAIEmbedderConfig = Omit<EmbeddingProviderConfig, "provider">;

export function createOpenAICompatibleEmbedder(config: OpenAIEmbedderConfig): EmbeddingProvider {
  const base: StandardEmbeddingConfig = {
    baseURL: config.baseURL,
    apiKey: config.apiKey,
    model: config.model,
    dims: config.dims,
  };
  const maxBatchSize = config.maxBatchSize ?? 32;

  return {
    name: "openai-compatible",
    dims: config.dims,
    supportsSparse: false,
    async embed(texts) {
      if (texts.length === 0) return [];
      const out: EmbeddingResult[] = [];
      for (let i = 0; i < texts.length; i += maxBatchSize) {
        out.push(...(await embedDenseBatch(base, texts.slice(i, i + maxBatchSize))));
      }
      return out;
    },
  };
}
