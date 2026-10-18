// 实体库 upsert：归一化精确匹配优先，其次语义相似度 ≥0.95 合并（setPayload 追加关联），
// 否则新建。容错原则：单实体处理失败仅告警，不阻塞写入主流程。
import { createHash, randomUUID } from "node:crypto";
import type { EmbeddingProvider } from "../embedding";
import type { VectorStore } from "../store";

export const MERGE_SIMILARITY = 0.95;

const normalize = (text: string) => text.trim().toLowerCase();

export interface EntityUpsertInput {
  items: Array<{ text: string; memoryId: string }>;
  userId: string;
  store: VectorStore;
  embedder: EmbeddingProvider;
}

/** text 为 normalize 后的实体名（实体点 payload.data） */
export interface EntityAssignment {
  text: string;
  entityId: string;
  /** 全部关联的记忆 id（含本次与历史写入） */
  memoryIds: string[];
}

export async function upsertEntities(input: EntityUpsertInput): Promise<EntityAssignment[]> {
  // 按 normalize 后的文本聚合：同一次写入中多个记忆涉及同一实体时只判断一次，
  // 并关联全部 memoryId
  const byText = new Map<string, Set<string>>();
  for (const { text, memoryId } of input.items) {
    const key = normalize(text);
    if (!key) continue;
    if (!byText.has(key)) byText.set(key, new Set());
    byText.get(key)!.add(memoryId);
  }

  const assignments: EntityAssignment[] = [];

  for (const [key, memoryIds] of byText) {
    try {
      const [vec] = await input.embedder.embed([key]);
      const hits = await input.store.searchDense(vec.dense, 3, {
        userId: input.userId,
      });

      const exact = hits.find((h) => normalize(h.payload.data) === key);
      const semantic = hits.length > 0 && hits[0].score >= MERGE_SIMILARITY ? hits[0] : undefined;
      const target = exact ?? semantic;

      if (target) {
        const merged = [...new Set([...(target.payload.linkedMemoryIds ?? []), ...memoryIds])];
        await input.store.setPayload(target.id, { linkedMemoryIds: merged });
        assignments.push({
          text: key,
          entityId: target.id,
          memoryIds: merged,
        });
      } else {
        const entityId = randomUUID();
        await input.store.upsert([
          {
            id: entityId,
            vector: {
              dense: vec.dense,
              ...(vec.sparse ? { sparse: vec.sparse } : {}),
            },
            payload: {
              data: key,
              hash: createHash("md5").update(key).digest("hex"),
              linkedMemoryIds: [...memoryIds],
              userId: input.userId,
              createdAt: new Date().toISOString(),
            },
          },
        ]);
        assignments.push({
          text: key,
          entityId,
          memoryIds: [...memoryIds],
        });
      }
    } catch (e) {
      console.warn(`实体 "${key}" 处理失败，跳过:`, e);
    }
  }

  return assignments;
}
