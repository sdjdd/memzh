export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ExistingMemory {
  id: string;
  text: string;
}

export interface ExtractedMemory {
  text: string;
  attributedTo: "user" | "assistant";
  /** 对话原文中的具体名词，≤8 个 */
  entities?: string[];
}

export interface OldMemoryDecision {
  id: string;
  event: "NONE" | "DELETE";
}

export interface FactDecision {
  fact: string;
  event: "ADD" | "NONE" | "UPDATE";
  /** UPDATE 的目标记忆 id */
  targetId?: string;
  /** UPDATE 后的新文本 */
  newText?: string;
}

export interface MemoryDecision {
  oldMemories: OldMemoryDecision[];
  facts: FactDecision[];
}

export interface MemoryEvent {
  event: "ADD" | "UPDATE" | "DELETE" | "NONE" | "DEDUP";
  /** ADD: 新记忆 id；UPDATE/DELETE: 目标记忆 id */
  id?: string;
  /** ADD/UPDATE: 最终文本；DELETE: 被删文本；NONE/DEDUP: 原事实 */
  text: string;
  /** UPDATE 的旧文本 */
  previousText?: string;
}

export interface ExtractOptions {
  existingMemories?: ExistingMemory[];
  lastKMessages?: ChatMessage[];
  observationDate?: string;
  customInstructions?: string;
}

export interface SparseVector {
  indices: number[];
  values: number[];
}

export interface EmbeddingResult {
  dense: number[];
  sparse?: SparseVector;
}

export interface VectorPayload {
  data: string;
  hash?: string;
  createdAt?: string;
  updatedAt?: string;
  attributedTo?: "user" | "assistant";
  userId?: string;
  /** 记忆点专用：该记忆关联的实体点 id */
  entityIds?: string[];
  /** 实体点专用：该实体关联的记忆 id */
  linkedMemoryIds?: string[];
  [key: string]: unknown;
}

export interface VectorHit {
  id: string;
  score: number;
  payload: VectorPayload;
}

export interface PayloadPoint {
  id: string;
  payload: VectorPayload;
}

export interface VectorPoint {
  id: string;
  vector: { dense: number[]; sparse?: SparseVector };
  payload: VectorPayload;
}

export interface VectorFilter {
  userId?: string;
}
