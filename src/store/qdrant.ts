import { QdrantClient } from "@qdrant/js-client-rest";
import type {
  PayloadPoint,
  SparseVector,
  VectorFilter,
  VectorHit,
  VectorPoint,
  VectorPayload,
} from "../types";
import type { VectorStore } from "./index";

export interface QdrantStoreConfig {
  client: QdrantClient;
  collection: string;
  dims: number;
  sparse?: boolean;
}

export class QdrantStore implements VectorStore {
  private client: QdrantClient;
  private readonly sparse: boolean;

  constructor(private opts: QdrantStoreConfig) {
    this.client = opts.client;
    this.sparse = opts.sparse ?? true;
  }

  async ensureCollection(): Promise<void> {
    const exists = await this.client.collectionExists(this.opts.collection);
    if (exists.exists) return;
    // sparse 槽位存储模型原生的学习型稀疏权重，并非词频，不叠加服务端 IDF
    // modifier（相似度口径为纯点积）；若恢复本地 TF 稀疏，需重建集合并添加
    // modifier: "idf"。
    await this.client.createCollection(this.opts.collection, {
      vectors: { dense: { size: this.opts.dims, distance: "Cosine" } },
      ...(this.sparse ? { sparse_vectors: { sparse: {} } } : {}),
    });
    // userId 过滤依赖 payload 索引，数据规模增大后过滤仍保持 O(log n)
    await this.client.createPayloadIndex(this.opts.collection, {
      field_name: "userId",
      field_schema: "keyword",
    });
  }

  async upsert(points: VectorPoint[]): Promise<void> {
    if (points.length === 0) return;
    await this.client.upsert(this.opts.collection, {
      wait: true, // 写后立即可见（测试与一致性优先；高吞吐场景可关闭）
      points: points.map((p) => ({
        id: p.id,
        vector: p.vector.sparse
          ? { dense: p.vector.dense, sparse: p.vector.sparse }
          : { dense: p.vector.dense },
        payload: p.payload,
      })),
    });
  }

  private buildFilter(filter?: VectorFilter) {
    return filter?.userId
      ? { must: [{ key: "userId", match: { value: filter.userId } }] }
      : undefined;
  }

  async searchDense(vector: number[], topK: number, filter?: VectorFilter): Promise<VectorHit[]> {
    const res = await this.client.query(this.opts.collection, {
      query: vector,
      using: "dense",
      limit: topK,
      with_payload: true,
      filter: this.buildFilter(filter),
    });
    return res.points.map(toHit);
  }

  async searchSparse(
    vector: SparseVector,
    topK: number,
    filter?: VectorFilter,
  ): Promise<VectorHit[] | null> {
    try {
      const res = await this.client.query(this.opts.collection, {
        query: vector,
        using: "sparse",
        limit: topK,
        with_payload: true,
        filter: this.buildFilter(filter),
      });
      return res.points.map(toHit);
    } catch (e) {
      console.warn("稀疏检索失败，退化为纯语义:", e);
      return null;
    }
  }

  async delete(ids: string[]): Promise<void> {
    if (ids.length === 0) return;
    await this.client.delete(this.opts.collection, { points: ids, wait: true });
  }

  async getPoints(ids: string[]): Promise<PayloadPoint[]> {
    if (ids.length === 0) return [];
    const res = await this.client.retrieve(this.opts.collection, {
      ids,
      with_payload: true,
      with_vector: false,
    });
    return res.map((p) => ({
      id: String(p.id),
      payload: p.payload as VectorPayload,
    }));
  }

  async setPayload(id: string, patch: Partial<VectorPayload>): Promise<void> {
    await this.client.setPayload(this.opts.collection, {
      payload: patch,
      points: [id],
    });
  }
}

function toHit(p: any): VectorHit {
  return {
    id: String(p.id),
    score: p.score,
    payload: p.payload as VectorPayload,
  };
}
