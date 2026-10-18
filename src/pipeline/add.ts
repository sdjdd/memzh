import { createHash, randomUUID } from "node:crypto";
import type { EmbeddingProvider } from "../embedding";
import type { MemoryExtractor } from "../extraction";
import type { MemoryDecider } from "./decide";
import type { VectorStore } from "../store";
import type {
  ChatMessage,
  ExistingMemory,
  ExtractedMemory,
  MemoryEvent,
  VectorHit,
  VectorPoint,
  VectorPayload,
} from "../types";
import { upsertEntities } from "./entities";

export interface AddOptions {
  userId: string;
  observationDate?: string;
  customInstructions?: string;
  existingTopK?: number;
}

export interface AddResult {
  extracted: ExtractedMemory[];
  /** 本次新建的记忆（UPDATE 原地覆写不产生新 id，见 events） */
  stored: Array<{ id: string; text: string }>;
  dedupedExact: number;
  events: MemoryEvent[];
}

export interface PipelineDeps {
  store: VectorStore;
  entityStore: VectorStore;
  embedder: EmbeddingProvider;
  extractor: MemoryExtractor;
  decider: MemoryDecider;
}

const FACT_CANDIDATE_TOPK = 5;
const MAX_CANDIDATES = 20;

const md5 = (text: string) => createHash("md5").update(text).digest("hex");

export async function addMemories(
  messages: ChatMessage[],
  deps: PipelineDeps,
  opts: AddOptions,
): Promise<AddResult> {
  const { store, entityStore, embedder, extractor, decider } = deps;
  const { userId } = opts;
  const topK = opts.existingTopK ?? 10;
  const events: MemoryEvent[] = [];

  // Phase 1: 会话级检索
  const parsed = messages.map((m) => `${m.role}: ${m.content}`).join("\n");
  const [queryVec] = await embedder.embed([parsed]);
  const neighbors = await store.searchDense(queryVec.dense, topK, { userId });
  const existingMemories: ExistingMemory[] = neighbors.map((n) => ({
    id: n.id,
    text: n.payload.data,
  }));
  const existingHashes = new Set(neighbors.map((n) => n.payload.hash).filter(Boolean));

  // Phase 2: LLM 抽取
  const extracted = await extractor.extract(messages, {
    existingMemories,
    observationDate: opts.observationDate,
    customInstructions: opts.customInstructions,
  });
  if (extracted.length === 0) {
    return { extracted, stored: [], dedupedExact: 0, events };
  }

  // Phase 3: 批量嵌入事实文本（向量复用于 Phase 5 候选检索与 Phase 7 写入，零额外嵌入）
  const texts = extracted.map((m) => m.text);
  const vectors = await embedder.embed(texts);

  // Phase 4: MD5 精确去重
  const seenHashes = new Set<string>();
  const survivors: number[] = [];
  let dedupedExact = 0;
  for (let i = 0; i < texts.length; i++) {
    const hash = md5(texts[i]);
    if (existingHashes.has(hash) || seenHashes.has(hash)) {
      dedupedExact++;
      events.push({ event: "DEDUP", text: texts[i] });
      continue;
    }
    seenHashes.add(hash);
    survivors.push(i);
  }
  if (survivors.length === 0) {
    return { extracted, stored: [], dedupedExact, events };
  }

  // Phase 5: 决策候选池 = 会话级近邻 ∪ 逐事实近邻（复用事实向量）
  // 仅靠会话级查询时，个别事实的目标旧记忆可能落不到 topK 内
  const candidateMap = new Map<string, VectorHit>();
  for (const n of neighbors) candidateMap.set(n.id, n);
  for (const i of survivors) {
    const hits = await store.searchDense(vectors[i].dense, FACT_CANDIDATE_TOPK, {
      userId,
    });
    for (const h of hits) {
      if (!candidateMap.has(h.id)) candidateMap.set(h.id, h);
    }
  }
  const candidates = [...candidateMap.values()].slice(0, MAX_CANDIDATES);
  const candidateById = new Map(candidates.map((c) => [c.id, c]));

  // Phase 6: 变更决策（单次 LLM 调用，批量对比）
  const decision = await decider.decide(
    survivors.map((i) => texts[i]),
    candidates.map((c) => ({ id: c.id, text: c.payload.data })),
  );

  // Phase 7: 消化决策结果
  // 旧行只有 NONE/DELETE 两种，先处理 DELETE，
  // 之后事实行遇到指向已删记忆的 UPDATE 时退化为 ADD
  const deletedIds = new Set<string>();
  for (const old of decision.oldMemories) {
    const hit = candidateById.get(old.id);
    if (!hit) {
      console.warn(`决策返回了候选集之外的 id，忽略: ${old.id}`);
      continue;
    }
    if (old.event === "DELETE") {
      deletedIds.add(old.id);
      events.push({ event: "DELETE", id: old.id, text: hit.payload.data });
    }
  }

  // LLM 漏行的事实默认 ADD（宁可多存）
  const mentionedFacts = new Set<string>();
  const addIndices = new Set<number>();
  /** 同一 id 被多条事实 UPDATE 时，后写覆盖先写 */
  const updateById = new Map<string, { point: VectorPoint; factIdx: number }>();
  for (const f of decision.facts) {
    mentionedFacts.add(f.fact);
    const idx = survivors.find((i) => texts[i] === f.fact);
    if (idx === undefined) continue; // LLM 编造的事实，忽略
    if (f.event === "NONE") {
      events.push({ event: "NONE", text: f.fact });
      continue;
    }
    if (f.event === "UPDATE") {
      const target = f.targetId ? candidateById.get(f.targetId) : undefined;
      const newText = f.newText?.trim();
      if (!target || !newText || deletedIds.has(target.id)) {
        // 目标缺失/新文本缺失/目标已被本次决策删除：退化为 ADD，保住新事实不丢
        console.warn(`UPDATE 决策无法落地（目标或 new_text 缺失），退化为 ADD: ${f.fact}`);
        addIndices.add(idx);
        continue;
      }
      const [vec] = await embedder.embed([newText]);
      updateById.set(target.id, {
        factIdx: idx,
        point: {
          id: target.id,
          vector: {
            dense: vec.dense,
            ...(vec.sparse ? { sparse: vec.sparse } : {}),
          },
          payload: {
            ...target.payload,
            data: newText,
            hash: md5(newText),
            updatedAt: new Date().toISOString(),
          },
        },
      });
      events.push({
        event: "UPDATE",
        id: target.id,
        text: newText,
        previousText: target.payload.data,
      });
      continue;
    }
    addIndices.add(idx);
  }
  for (const i of survivors) {
    if (!mentionedFacts.has(texts[i])) addIndices.add(i);
  }
  const updatePoints = [...updateById.values()];

  // Phase 8: 执行
  // DELETE：删点 + 从 payload.entityIds 反查实体点，摘除死链（辅助通道，失败不阻塞）
  for (const id of deletedIds) {
    const entityIds = candidateById.get(id)!.payload.entityIds ?? [];
    try {
      if (entityIds.length > 0) {
        const entityPoints = await entityStore.getPoints(entityIds);
        for (const p of entityPoints) {
          const linked = (p.payload.linkedMemoryIds ?? []).filter((x) => x !== id);
          await entityStore.setPayload(p.id, { linkedMemoryIds: linked });
        }
      }
      await store.delete([id]);
    } catch (e) {
      console.warn(`记忆 ${id} 删除或回链清理失败:`, e);
    }
  }

  // ADD：新点（向量复用 Phase 3 结果）
  const now = new Date().toISOString();
  const newRecords = [...addIndices].map((i) => {
    const id = randomUUID();
    const payload: VectorPayload = {
      data: texts[i],
      hash: md5(texts[i]),
      createdAt: now,
      attributedTo: extracted[i].attributedTo,
      userId,
    };
    return {
      id,
      text: texts[i],
      entities: extracted[i].entities ?? [],
      point: {
        id,
        vector: {
          dense: vectors[i].dense,
          ...(vectors[i].sparse ? { sparse: vectors[i].sparse } : {}),
        },
        payload,
      } satisfies VectorPoint,
    };
  });

  const allPoints = [...updatePoints.map((u) => u.point), ...newRecords.map((r) => r.point)];
  if (allPoints.length > 0) {
    await store.upsert(allPoints);
    for (const r of newRecords) {
      events.push({ event: "ADD", id: r.id, text: r.text });
    }
  }

  // 实体库 upsert（ADD 与 UPDATE 都要挂实体关联），
  // 然后把实体点 id 回写到记忆 payload.entityIds。
  // UPDATE 的实体取自触发它的那条事实（UPDATE 行自带 fact 对应的抽取条目）
  const entitySources = [
    ...newRecords.map((r) => ({ id: r.id, entities: r.entities })),
    ...updatePoints.map((u) => ({
      id: u.point.id,
      entities: extracted[u.factIdx]?.entities ?? [],
    })),
  ];
  const entityItems = entitySources.flatMap((r) =>
    r.entities.slice(0, 8).map((text) => ({ text, memoryId: r.id })),
  );
  if (entityItems.length > 0) {
    try {
      const assignments = await upsertEntities({
        items: entityItems,
        userId,
        store: entityStore,
        embedder,
      });
      const byMemory = new Map<string, Set<string>>();
      for (const a of assignments) {
        for (const memoryId of a.memoryIds) {
          // 只回写本次涉及的记忆；其他记忆的关联由历史写入维护
          const involved =
            newRecords.some((r) => r.id === memoryId) ||
            updatePoints.some((u) => u.point.id === memoryId);
          if (!involved) continue;
          if (!byMemory.has(memoryId)) byMemory.set(memoryId, new Set());
          byMemory.get(memoryId)!.add(a.entityId);
        }
      }
      for (const [memoryId, entityIds] of byMemory) {
        try {
          // UPDATE：与旧回链合并；ADD：直接写
          const prev = candidateById.get(memoryId)?.payload.entityIds ?? [];
          await store.setPayload(memoryId, {
            entityIds: [...new Set([...prev, ...entityIds])],
          });
        } catch (e) {
          console.warn(`记忆 ${memoryId} 的实体回链写入失败:`, e);
        }
      }
    } catch (e) {
      console.warn("实体库 upsert 失败，跳过（记忆本身已写入）:", e);
    }
  }

  return {
    extracted,
    stored: newRecords.map((r) => ({ id: r.id, text: r.text })),
    dedupedExact,
    events,
  };
}
