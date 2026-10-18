import type {
  PayloadPoint,
  SparseVector,
  VectorFilter,
  VectorHit,
  VectorPoint,
  VectorPayload,
} from "../types";

export interface VectorStore {
  /** 创建集合（含 payload 索引） */
  ensureCollection(): Promise<void>;
  /** 缺少 sparse 的点仅写 dense */
  upsert(points: VectorPoint[]): Promise<void>;
  /** 不存在的 id 静默忽略 */
  delete(ids: string[]): Promise<void>;
  /** 集合中不存在的 id 不出现在返回里 */
  getPoints(ids: string[]): Promise<PayloadPoint[]>;
  /** 余弦相似度 */
  searchDense(vector: number[], topK: number, filter?: VectorFilter): Promise<VectorHit[]>;
  /**
   * 点积相似度。返回 null 表示通道不可用（不支持稀疏或检索失败），区别于无命中
   */
  searchSparse(
    vector: SparseVector,
    topK: number,
    filter?: VectorFilter,
  ): Promise<VectorHit[] | null>;
  /** 部分合并 payload，不修改向量 */
  setPayload(id: string, patch: Partial<VectorPayload>): Promise<void>;
}

export type { PayloadPoint, SparseVector, VectorFilter, VectorHit, VectorPoint, VectorPayload };
