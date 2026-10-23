// 评测：fetch 阶段重建语料并缓存三通道取数（调 API，一次）；
// sweep 阶段读取缓存，离线扫 fuseChannels 参数网格（零 API 成本）；
// reliability 阶段对语料做 N 轮纯抽取（不落库），度量空抽取率与实体漏标率。
// 评测卫生：查询分 dev / heldout 两段——参数网格只在 dev 上选参，
// heldout 仅报告不参与选参；提示词示例同样不得触碰两段的词汇（见 hygiene 单测）。
// 用法：npx tsx eval/run.ts [fetch | sweep | all | reliability [runs]]
import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { QdrantClient } from "@qdrant/js-client-rest";
import {
  createEmbedder,
  createExtractor,
  createMemorySystem,
  fetchChannels,
  QdrantStore,
  type ScoringParams,
} from "../src/index";
import { EXTRACTION_SYSTEM_PROMPT } from "../src/extraction/prompt";
import { DECISION_SYSTEM_PROMPT } from "../src/pipeline/decide";
import type { ChatMessage } from "../src/types";
import { loadEnv } from "../src/env";
import {
  containsAny,
  evaluate,
  INTENTS,
  matchesTerm,
  splitOf,
  type CacheEntry,
  type EvalQuery,
  type Metrics,
} from "./metrics";

loadEnv();

const DIR = path.dirname(fileURLToPath(import.meta.url));
const QDRANT_URL = "http://127.0.0.1:6333";
const MEMORY_COLLECTION = "zh_memories_eval";
const ENTITY_COLLECTION = "zh_entities_eval";
const DIMS = 2048;
const USER_ID = "eval_user";
/** 判定窗口：hit@1 / hit@3 / MRR 均在前 5 条内计算 */
const JUDGE_TOPK = 5;

interface Corpus {
  conversations: {
    id: string;
    date: string;
    /** 实体对标的：这些名词应出现在该对话抽出记忆的 entities 里（度量实体漏标） */
    expectEntities?: string[];
    messages: ChatMessage[];
  }[];
}
interface Cache {
  createdAt: string;
  corpusHash: string;
  queriesHash: string;
  /** 抽取/决策 prompt 与模型名的指纹，sweep 时校验缓存是否代表当前管线 */
  pipelineHash: string;
  memoryCount: number;
  entries: CacheEntry[];
}

interface Row extends Metrics {
  threshold: number;
  mid: number;
  steep: number;
  minSim: number;
  score: number;
}

const LLM_MODEL = "deepseek-v4.1-flash";
const EMBED_MODEL = "doubao-embedding-vision";

const provider = createOpenAICompatible({
  name: "my-provider",
  baseURL: process.env.LLM_BASE_URL!,
  apiKey: process.env.LLM_API_KEY!,
});

const qdrant = new QdrantClient({
  url: QDRANT_URL,
});

const store = new QdrantStore({
  client: qdrant,
  collection: MEMORY_COLLECTION,
  dims: DIMS,
});
const entityStore = new QdrantStore({
  client: qdrant,
  collection: ENTITY_COLLECTION,
  dims: DIMS,
});
const embedder = createEmbedder({
  provider: "volcengine",
  apiKey: process.env.EMBEDDING_API_KEY!,
  baseURL: process.env.EMBEDDING_BASE_URL!,
  model: EMBED_MODEL,
  dims: DIMS,
  sparse: true,
});

const deps = { store, entityStore, embedder };

function md5(s: string): string {
  return createHash("md5").update(s).digest("hex");
}

function readFile(rel: string): string {
  return fs.readFileSync(path.join(DIR, rel), "utf8");
}

// 缓存新鲜度指纹：抽取/决策 prompt 或模型变更后，旧缓存的三通道取数不再代表当前管线
const PIPELINE_HASH = md5(
  EXTRACTION_SYSTEM_PROMPT + DECISION_SYSTEM_PROMPT + LLM_MODEL + EMBED_MODEL,
);

async function fetchPhase(): Promise<void> {
  const corpus: Corpus = JSON.parse(readFile("corpus.json"));
  const queries: { queries: EvalQuery[] } = JSON.parse(readFile("queries.json"));

  // 评测专用集合整体重建；实体集合必须一并删除，否则残留指向已删记忆的无效关联
  await qdrant.deleteCollection(MEMORY_COLLECTION).catch(() => {});
  await qdrant.deleteCollection(ENTITY_COLLECTION).catch(() => {});

  // 先删后建：createMemorySystem 组装时 ensure 集合，必须晚于上面的删除
  const memory = await createMemorySystem({
    embedder,
    model: provider(LLM_MODEL),
    store,
    entityStore,
  });

  console.log("===== 写入语料 =====");
  for (const conv of corpus.conversations) {
    const r = await memory.add(conv.messages, {
      userId: USER_ID,
      observationDate: conv.date,
    });
    console.log(`[${conv.id} ${conv.date}] 入库 ${r.stored.length} 条`);
    for (const s of r.stored) console.log(`   + ${s.text}`);
  }

  const stored = await qdrant.scroll(MEMORY_COLLECTION, {
    limit: 100,
    with_payload: true,
  });
  const texts = stored.points.map((p) => String((p.payload as { data?: string }).data ?? ""));
  console.log(`\n语料库共 ${texts.length} 条记忆`);

  console.log("\n===== fetchChannels 缓存 =====");
  const entries: CacheEntry[] = [];
  for (const q of queries.queries) {
    const fetches = await fetchChannels(q.q, deps, {
      userId: USER_ID,
      topK: 15,
    });
    const covered = q.expect.length === 0 || texts.some((t) => containsAny(t, q.expect));
    entries.push({ ...q, covered, fetches });
    const bm25Info = fetches.bm25 ? String(fetches.bm25.length) : "不可用";
    console.log(
      `[${q.id}${splitOf(q) === "heldout" ? "*" : ""}] ${q.q}  bm25=${bm25Info} 实体=${fetches.entityMatches.length}` +
        (covered ? "" : "  !! 语料未覆盖"),
    );
    // 每条 query 间隔发送，避免持续高频触发账号限流
    await new Promise((resolve) => setTimeout(resolve, 300));
  }

  const cache: Cache = {
    createdAt: new Date().toISOString(),
    corpusHash: md5(readFile("corpus.json")),
    queriesHash: md5(readFile("queries.json")),
    pipelineHash: PIPELINE_HASH,
    memoryCount: texts.length,
    entries,
  };
  fs.writeFileSync(path.join(DIR, "cache.json"), JSON.stringify(cache, null, 2));
  console.log(`\n缓存已写入 cache.json（${entries.length} 条 query）`);
}

function fmt(m: Metrics): string {
  return `hit@1=${m.hit1.toFixed(3)} hit@3=${m.hit3.toFixed(3)} mrr=${m.mrr.toFixed(3)} fp=${m.fp.toFixed(3)}`;
}

function pct(sorted: number[], p: number): number {
  return sorted.length ? sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))] : 0;
}

async function sweepPhase(): Promise<void> {
  const cache: Cache = JSON.parse(readFile("cache.json"));
  if (
    cache.corpusHash !== md5(readFile("corpus.json")) ||
    cache.queriesHash !== md5(readFile("queries.json")) ||
    cache.pipelineHash !== PIPELINE_HASH
  ) {
    console.warn("!! 语料/查询或管线指纹与缓存不一致，结论可能过期，建议重跑 fetch");
  }
  const entries = cache.entries;

  const uncovered = entries.filter((e) => !e.covered);
  if (uncovered.length > 0) {
    console.log("\n===== 语料未覆盖（抽取缺失，不计入检索指标）=====");
    for (const e of uncovered)
      console.log(`  [${e.id}] ${e.q}  expect=${JSON.stringify(e.expect)}`);
  }
  const judged = entries.filter((e) => e.covered);
  const devEntries = judged.filter((e) => splitOf(e) === "dev");
  const heldoutEntries = judged.filter((e) => splitOf(e) === "heldout");

  console.log("\n===== 缓存分布 =====");
  const bm25Raw = entries
    .flatMap((e) => (e.fetches.bm25 ?? []).map((h) => h.score ?? 0))
    .filter((s) => s > 0)
    .sort((a, b) => a - b);
  console.log(
    `bm25 原始分（>0）: n=${bm25Raw.length}  p50=${pct(bm25Raw, 0.5).toFixed(3)}` +
      `  p90=${pct(bm25Raw, 0.9).toFixed(3)}  max=${pct(bm25Raw, 1).toFixed(3)}`,
  );
  for (const e of entries.filter((x) => x.intent === "abstention")) {
    const top = Math.max(0, ...e.fetches.semantic.map((h) => h.score ?? 0));
    console.log(`  噪声地板 [${e.id}] ${e.q}  top语义=${top.toFixed(3)}`);
  }

  console.log("\n===== 基线（DEFAULT_SCORING + 自适应 BM25） =====");
  console.log(`全体: ${fmt(evaluate(judged, {}, JUDGE_TOPK))}`);
  for (const [label, subset] of [
    ["dev", devEntries],
    ["heldout", heldoutEntries],
  ] as const) {
    console.log(`  ${label}: ${fmt(evaluate(subset, {}, JUDGE_TOPK))} (n=${subset.length})`);
    for (const intent of INTENTS) {
      const sub = subset.filter((e) => e.intent === intent);
      if (sub.length > 0)
        console.log(`    ${intent}: ${fmt(evaluate(sub, {}, JUDGE_TOPK))} (n=${sub.length})`);
    }
  }

  // 网格范围参考上方打印的 bm25 原始分分布设定
  const THRESHOLDS = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45];
  const MIDPOINTS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2];
  const STEEPNESS = [0.5, 1, 2];
  const MINSIMS = [0.5, 0.6, 0.7];

  const rows: Row[] = [];
  for (const threshold of THRESHOLDS) {
    for (const mid of MIDPOINTS) {
      for (const steep of STEEPNESS) {
        for (const minSim of MINSIMS) {
          const m = evaluate(
            devEntries,
            {
              threshold,
              bm25Midpoint: mid,
              bm25Steepness: steep,
              entityMinSim: minSim,
            },
            JUDGE_TOPK,
          );
          rows.push({
            threshold,
            mid,
            steep,
            minSim,
            ...m,
            score: m.hit3 - 2 * m.fp,
          });
        }
      }
    }
  }
  rows.sort((a, b) => b.score - a.score || b.hit1 - a.hit1 || b.mrr - a.mrr);

  console.log("\n===== 网格 top 15（score = hit@3 - 2×误报率） =====");
  console.log("thr    mid    steep  minSim hit@1  hit@3  mrr    fp     score");
  for (const r of rows.slice(0, 15)) {
    console.log(
      `${r.threshold.toFixed(2)}   ${r.mid.toFixed(2)}   ${r.steep.toFixed(1)}    ${r.minSim.toFixed(2)}   ` +
        `${r.hit1.toFixed(3)}  ${r.hit3.toFixed(3)}  ${r.mrr.toFixed(3)}  ${r.fp.toFixed(3)}  ${r.score.toFixed(3)}`,
    );
  }

  const best = rows[0];
  const bestNoFp = rows.find((r) => r.fp === 0) ?? best;
  const overrides: Partial<ScoringParams> = {
    threshold: best.threshold,
    bm25Midpoint: best.mid,
    bm25Steepness: best.steep,
    entityMinSim: best.minSim,
  };
  console.log(
    `\n===== 最优参数逐意图细分（dev 网格选出：thr=${best.threshold} mid=${best.mid} steep=${best.steep} minSim=${best.minSim}） =====`,
  );
  for (const intent of INTENTS) {
    const sub = devEntries.filter((e) => e.intent === intent);
    if (sub.length > 0)
      console.log(`  ${intent}: ${fmt(evaluate(sub, overrides, JUDGE_TOPK))} (n=${sub.length})`);
  }

  console.log(
    `\n===== held-out（不参与网格选参，仅报告：thr=${best.threshold} mid=${best.mid} steep=${best.steep} minSim=${best.minSim}） =====`,
  );
  console.log(`  @默认参数: ${fmt(evaluate(heldoutEntries, {}, JUDGE_TOPK))}`);
  console.log(`  @最优参数: ${fmt(evaluate(heldoutEntries, overrides, JUDGE_TOPK))}`);
  for (const intent of INTENTS) {
    const sub = heldoutEntries.filter((e) => e.intent === intent);
    if (sub.length > 0)
      console.log(`    ${intent}: ${fmt(evaluate(sub, overrides, JUDGE_TOPK))} (n=${sub.length})`);
  }
  console.log(
    `\n误报为零的最优: thr=${bestNoFp.threshold} mid=${bestNoFp.mid} steep=${bestNoFp.steep} minSim=${bestNoFp.minSim}  ${fmt(bestNoFp)}`,
  );

  fs.writeFileSync(
    path.join(DIR, "report.json"),
    JSON.stringify(
      {
        createdAt: new Date().toISOString(),
        memoryCount: cache.memoryCount,
        cacheCreatedAt: cache.createdAt,
        baseline: evaluate(judged, {}, JUDGE_TOPK),
        dev: evaluate(devEntries, {}, JUDGE_TOPK),
        heldout: evaluate(heldoutEntries, {}, JUDGE_TOPK),
        heldoutAtBest: evaluate(heldoutEntries, overrides, JUDGE_TOPK),
        best,
        bestNoFp,
        top: rows.slice(0, 50),
      },
      null,
      2,
    ),
  );
  console.log("\n报告已写入 report.json");
}

/**
 * 抽取可靠性度量：N 轮纯抽取（跳过检索上下文与落库，隔离抽取环节本身）。
 * 抽取是概率行为，单次运行观察不到偶发失效；本命令给出空抽取率、
 * expect 关键词覆盖率分布与实体命中率的多次抽样估计。
 */
async function reliabilityPhase(runs: number): Promise<void> {
  const corpus: Corpus = JSON.parse(readFile("corpus.json"));
  const queries: { queries: EvalQuery[] } = JSON.parse(readFile("queries.json"));
  const extractor = createExtractor(provider(LLM_MODEL));

  const convCount = corpus.conversations.length;
  let totalItems = 0;
  const emptyEvents: string[] = [];
  const queryCover = new Map<string, number>(
    queries.queries.filter((q) => q.intent !== "abstention").map((q) => [q.id, 0]),
  );
  const entityCover = new Map<string, number>();

  for (let r = 1; r <= runs; r++) {
    console.log(`----- 第 ${r}/${runs} 轮 -----`);
    const texts: string[] = [];
    for (const conv of corpus.conversations) {
      const items = await extractor.extract(conv.messages, {
        observationDate: conv.date,
      });
      totalItems += items.length;
      if (items.length === 0) {
        emptyEvents.push(`r${r}/${conv.id}`);
        console.log(`  !! [${conv.id}] 空抽取`);
      }
      for (const s of items) texts.push(s.text);

      const expect = conv.expectEntities ?? [];
      if (expect.length > 0) {
        const hit = expect.filter((exp) =>
          items.some((m) => (m.entities ?? []).some((e) => matchesTerm(e, exp))),
        );
        if (hit.length === expect.length) {
          entityCover.set(conv.id, (entityCover.get(conv.id) ?? 0) + 1);
        } else {
          const missing = expect.filter((e) => !hit.includes(e));
          console.log(`  ~~ [${conv.id}] 实体漏标: ${missing.join("、")}`);
        }
      }
      // 轮次内间隔发送，降低触发限流的概率
      await new Promise((resolve) => setTimeout(resolve, 200));
    }
    for (const q of queries.queries) {
      // 弃答查询的 expect 为空，文本覆盖率语义不适用，不计入
      if (q.intent === "abstention") continue;
      if (texts.some((t) => containsAny(t, q.expect))) {
        queryCover.set(q.id, (queryCover.get(q.id) ?? 0) + 1);
      }
    }
  }

  console.log(`\n===== 抽取可靠性汇总（${runs} 轮 × ${convCount} 段，纯抽取无检索上下文） =====`);
  const sampleCount = runs * convCount;
  console.log(
    `空抽取: ${emptyEvents.length}/${sampleCount}` +
      (emptyEvents.length > 0 ? `  (${emptyEvents.join(" ")})` : ""),
  );
  console.log(`平均每段抽取条数: ${(totalItems / sampleCount).toFixed(2)}`);

  const meanCover = [...queryCover.values()].reduce((a, b) => a + b, 0) / queryCover.size / runs;
  console.log(`expect 覆盖率均值: ${(meanCover * 100).toFixed(1)}%`);
  const partialCover = [...queryCover.entries()].filter(([, c]) => c < runs);
  if (partialCover.length > 0) {
    console.log("未在全部轮次覆盖的 query:");
    for (const [id, c] of partialCover) {
      const q = queries.queries.find((x) => x.id === id)!;
      console.log(`  [${id}] ${c}/${runs}  ${q.q}`);
    }
  }

  const tracked = corpus.conversations.filter((c) => (c.expectEntities?.length ?? 0) > 0);
  if (tracked.length > 0) {
    const meanEntity =
      tracked.reduce((a, c) => a + (entityCover.get(c.id) ?? 0), 0) / tracked.length / runs;
    console.log(
      `实体命中率均值（对话内全部对标的命中才算命中）: ${(meanEntity * 100).toFixed(1)}%`,
    );
    for (const c of tracked.filter((x) => (entityCover.get(x.id) ?? 0) < runs)) {
      console.log(`  [${c.id}] ${entityCover.get(c.id) ?? 0}/${runs} 轮全命中`);
    }
  }
}

const cmd = process.argv[2] ?? "all";
if (cmd === "fetch") await fetchPhase();
else if (cmd === "sweep") await sweepPhase();
else if (cmd === "all") {
  await fetchPhase();
  await sweepPhase();
} else if (cmd === "reliability") {
  await reliabilityPhase(Number(process.argv[3] ?? 10));
} else {
  console.error("用法: npx tsx eval/run.ts [fetch|sweep|all|reliability [runs]]");
  process.exitCode = 1;
}
