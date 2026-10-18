import type { LanguageModel } from "ai";
import { createExtractor, type MemoryExtractor } from "./extraction";
import { resolveEmbedder, type EmbeddingProvider, type EmbeddingProviderConfig } from "./embedding";
import type { VectorStore } from "./store";
import { QdrantStore, type QdrantStoreConfig } from "./store/qdrant";
import { addMemories, type AddOptions, type AddResult } from "./pipeline/add";
import { createMemoryDecider, type MemoryDecider } from "./pipeline/decide";
import { searchMemories, type SearchOptions, type SearchResultItem } from "./pipeline/search";
import type { ScoringParams } from "./pipeline/scoring";
import type { ChatMessage } from "./types";

export type {
  ChatMessage,
  EmbeddingResult,
  ExistingMemory,
  ExtractedMemory,
  FactDecision,
  MemoryDecision,
  MemoryEvent,
  OldMemoryDecision,
  SparseVector,
  VectorFilter,
  VectorHit,
  VectorPayload,
  VectorPoint,
} from "./types";
export type { EmbeddingProvider, EmbeddingProviderConfig } from "./embedding";
export { createEmbedder, registerEmbeddingProvider } from "./embedding";
export { createExtractor, type MemoryExtractor } from "./extraction";
export type { PayloadPoint, VectorStore } from "./store";
export { QdrantStore, type QdrantStoreConfig } from "./store/qdrant";
export {
  DEFAULT_SCORING,
  computeEntityBoosts,
  getBm25Params,
  normalizeBm25,
  scoreAndRank,
  type ScoringParams,
} from "./pipeline/scoring";
export { fetchChannels, fuseChannels } from "./pipeline/search";
export { createMemoryDecider, DECISION_SYSTEM_PROMPT } from "./pipeline/decide";
export type { MemoryDecider } from "./pipeline/decide";
export type { AddOptions, AddResult } from "./pipeline/add";
export type { ChannelFetches, SearchOptions, SearchResultItem } from "./pipeline/search";

export type StoreInput = QdrantStoreConfig | VectorStore;

export interface MemorySystemConfig {
  embedder: EmbeddingProvider | EmbeddingProviderConfig;
  model: LanguageModel;
  store: StoreInput;
  entityStore: StoreInput;
  /** 系统级打分参数覆盖（单次 search 还可再覆盖） */
  scoring?: Partial<ScoringParams>;
}

export interface MemorySystem {
  add(messages: ChatMessage[], opts: AddOptions): Promise<AddResult>;
  search(query: string, opts: SearchOptions): Promise<SearchResultItem[]>;
}

function isVectorStore(x: StoreInput): x is VectorStore {
  return typeof (x as VectorStore).ensureCollection === "function";
}

function resolveStore(input: StoreInput): VectorStore {
  return isVectorStore(input) ? input : new QdrantStore(input);
}

export async function createMemorySystem(config: MemorySystemConfig): Promise<MemorySystem> {
  const embedder = resolveEmbedder(config.embedder);
  const extractor: MemoryExtractor = createExtractor(config.model);
  const decider: MemoryDecider = createMemoryDecider(config.model);
  const store = resolveStore(config.store);
  const entityStore = resolveStore(config.entityStore);
  const deps = { store, entityStore, embedder, extractor, decider };

  await store.ensureCollection();
  await entityStore.ensureCollection();

  return {
    add: (messages, opts) => addMemories(messages, deps, opts),
    search: (query, opts) =>
      searchMemories(query, deps, {
        ...opts,
        scoring: { ...config.scoring, ...opts.scoring },
      }),
  };
}
