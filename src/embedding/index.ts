import type { EmbeddingResult, SparseVector } from "../types";
import { createVolcengineEmbedder } from "./volcengine";
import { createOpenAICompatibleEmbedder } from "./openai";

export interface EmbeddingProvider {
  readonly name: string;
  readonly dims: number;
  /** 能力声明：是否可能返回稀疏向量；单次调用是否真带 sparse 以响应为准 */
  readonly supportsSparse: boolean;
  /**
   * 契约：
   * - 结果与输入逐条对应、顺序对齐
   * - sparse 缺省 = 本次没给出稀疏，调用方必须容忍
   * - dense 维度恒等于 dims，不符时实现应抛错
   */
  embed(texts: string[]): Promise<EmbeddingResult[]>;
}

export interface EmbeddingProviderConfig {
  /** 通过 registerEmbeddingProvider 注册的名字 */
  provider: string;
  apiKey: string;
  model: string;
  dims: number;
  baseURL: string;
  /** 请求模型原生稀疏向量（仅部分 provider 支持；不支持时实现内部降级纯 dense） */
  sparse?: boolean;
  /** 标准端点单次批量上限 */
  maxBatchSize?: number;
  /** 逐条调用时的并发上限 */
  concurrency?: number;
}

type EmbedderFactory = (config: EmbeddingProviderConfig) => EmbeddingProvider;

const registry = new Map<string, EmbedderFactory>();

export function registerEmbeddingProvider(name: string, factory: EmbedderFactory): void {
  registry.set(name, factory);
}

export function createEmbedder(config: EmbeddingProviderConfig): EmbeddingProvider {
  const factory = registry.get(config.provider);
  if (!factory) {
    throw new Error(
      `未知的 embedding provider: "${config.provider}"（已注册: ${[...registry.keys()].join(", ")}）`,
    );
  }
  return factory(config);
}

export function resolveEmbedder(
  input: EmbeddingProvider | EmbeddingProviderConfig,
): EmbeddingProvider {
  if ("embed" in input && typeof input.embed === "function") {
    return input as EmbeddingProvider;
  }
  return createEmbedder(input as EmbeddingProviderConfig);
}

registerEmbeddingProvider("volcengine", (cfg) => createVolcengineEmbedder(cfg));
registerEmbeddingProvider("openai-compatible", (cfg) => createOpenAICompatibleEmbedder(cfg));

export { createVolcengineEmbedder, createOpenAICompatibleEmbedder };
export type { EmbeddingResult, SparseVector };
