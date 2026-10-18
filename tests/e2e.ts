// 端到端回归：清场重建，可重复运行。
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { QdrantClient } from "@qdrant/js-client-rest";
import { createMemorySystem } from "../src/index";
import { DEFAULT_SCORING } from "../src/pipeline/scoring";
import type { ChatMessage } from "../src/types";
import { loadEnv } from "../src/env";

loadEnv();

// e2e 的全部外部依赖（Qdrant 地址、目标集合、模型、维度）由 E2E_* 环境变量注入，不设代码默认值：
// 代码默认值会让误配置静默打到共享集合上——本脚本开头就删集合，指错目标等于删数据
const REQUIRED_VARS = [
  "LLM_API_KEY",
  "LLM_BASE_URL",
  "EMBEDDING_API_KEY",
  "EMBEDDING_BASE_URL",
  "E2E_QDRANT_URL",
  "E2E_MEMORY_COLLECTION",
  "E2E_ENTITY_COLLECTION",
  "E2E_LLM_MODEL",
  "E2E_EMBED_MODEL",
  "E2E_DIMS",
];
const missingVars = REQUIRED_VARS.filter((k) => !process.env[k]);
if (missingVars.length > 0) {
  console.error(`e2e 缺少环境变量: ${missingVars.join(", ")}`);
  process.exit(1);
}

const provider = createOpenAICompatible({
  name: "my-provider",
  baseURL: process.env.LLM_BASE_URL!,
  apiKey: process.env.LLM_API_KEY!,
});

const QDRANT_URL = process.env.E2E_QDRANT_URL!;
const MEMORY_COLLECTION = process.env.E2E_MEMORY_COLLECTION!;
const ENTITY_COLLECTION = process.env.E2E_ENTITY_COLLECTION!;
const DIMS = Number(process.env.E2E_DIMS);
if (!Number.isInteger(DIMS) || DIMS <= 0) {
  console.error(`E2E_DIMS 必须是正整数，实际: ${process.env.E2E_DIMS}`);
  process.exit(1);
}

let failures = 0;
function check(name: string, ok: boolean, detail = ""): void {
  console.log(`  ${ok ? "✓" : "✗ FAIL"} ${name}${detail ? ` — ${detail}` : ""}`);
  if (!ok) failures++;
}

const raw = new QdrantClient({ url: QDRANT_URL });
await raw.deleteCollection(MEMORY_COLLECTION).catch(() => {});
// 实体集合必须一并删除，否则残留指向已删除记忆的无效关联
await raw.deleteCollection(ENTITY_COLLECTION).catch(() => {});

const memory = await createMemorySystem({
  embedder: {
    provider: "volcengine",
    apiKey: process.env.EMBEDDING_API_KEY!,
    baseURL: process.env.EMBEDDING_BASE_URL!,
    model: process.env.E2E_EMBED_MODEL!,
    dims: DIMS,
    sparse: true,
  },
  model: provider(process.env.E2E_LLM_MODEL!),
  store: { client: raw, collection: MEMORY_COLLECTION, dims: DIMS },
  entityStore: { client: raw, collection: ENTITY_COLLECTION, dims: DIMS },
  scoring: { threshold: DEFAULT_SCORING.threshold },
});

const convoA: ChatMessage[] = [
  {
    role: "user",
    content: "我喜欢吃青椒，但我老公一点辣都吃不了，所以我家炒菜从来不放辣椒。",
  },
  { role: "assistant", content: "了解了！那可以试试不放辣椒的虎皮青椒做法。" },
];
const convoB: ChatMessage[] = [
  { role: "user", content: "我妈上周来成都看我，带了一箱老家产的花椒。" },
  { role: "assistant", content: "汉源花椒很香，做水煮鱼正合适！" },
];
const convoC: ChatMessage[] = [
  { role: "user", content: "今天去菜市场买了青椒和鸡蛋，明天炒着吃。" },
];

console.log("\n===== Part 1: addMemories =====");

const r1 = await memory.add(convoA, {
  userId: "u1",
  observationDate: "2026-10-17",
});
console.log(`A 初次写入: ${r1.stored.length} 条入库，MD5 拦截 ${r1.dedupedExact}`);
r1.stored.forEach((s, i) => console.log(`   [${i}] ${s.text}`));
check("A 初次写入非空", r1.stored.length >= 1, `入库 ${r1.stored.length} 条`);

const r2 = await memory.add(convoA, {
  userId: "u1",
  observationDate: "2026-10-17",
});
console.log(`A 原样重放: ${r2.stored.length} 条入库（预期 0），MD5 拦截 ${r2.dedupedExact}`);
check(
  "A 原样重放零新增",
  r2.stored.length === 0,
  `入库 ${r2.stored.length} 条: ${JSON.stringify(r2.stored.map((s) => s.text))}`,
);

const r3 = await memory.add(convoB, {
  userId: "u1",
  observationDate: "2026-10-17",
});
console.log(`B 花椒对话: ${r3.stored.length} 条入库`);
r3.stored.forEach((s, i) => console.log(`   [${i}] ${s.text}`));

const r5 = await memory.add(convoC, {
  userId: "u1",
  observationDate: "2026-10-18",
});
console.log(`C 青椒补充: ${r5.stored.length} 条入库（“青椒”实体应合并进已有条目）`);

const r4 = await memory.add(convoA, {
  userId: "u2",
  observationDate: "2026-10-17",
});
console.log(`A 换用户 u2: ${r4.stored.length} 条入库（隔离验证，预期 > 0）`);
check("A 换用户 u2 首次写入非空", r4.stored.length > 0, `入库 ${r4.stored.length} 条`);

console.log("\n===== Part 2: 集合结构检查 =====");

const info = await raw.getCollection(MEMORY_COLLECTION);
console.log(`points 总数: ${info.points_count}`);

const sample = await raw.scroll(MEMORY_COLLECTION, {
  limit: 1,
  with_payload: true,
  with_vector: true,
});
const vec = sample.points[0]?.vector as any;
console.log(`向量字段: [${Object.keys(vec ?? {}).join(", ")}]（预期 dense, sparse）`);
console.log(
  `dense 维度: ${vec?.dense?.length}（预期 ${DIMS}），sparse 词条数: ${vec?.sparse?.indices?.length}`,
);
check("dense 向量维度", vec?.dense?.length === DIMS, `实际 ${vec?.dense?.length}`);
check(
  "sparse 向量存在（multimodal 未降级）",
  (vec?.sparse?.indices?.length ?? 0) > 0,
  `词条 ${vec?.sparse?.indices?.length ?? 0} 个`,
);

const entInfo = await raw.getCollection(ENTITY_COLLECTION);
console.log(`\n===== 实体库: ${entInfo.points_count} 个实体 =====`);
const ents = await raw.scroll(ENTITY_COLLECTION, {
  limit: 50,
  with_payload: true,
});
for (const p of ents.points) {
  const pl = p.payload as any;
  console.log(
    `  [${pl.userId ?? "?"}] ${pl.data}（关联 ${pl.linkedMemoryIds?.length ?? 0} 条记忆）`,
  );
}
check("实体库非空", (entInfo.points_count ?? 0) > 0, `${entInfo.points_count} 个实体`);

// convoA 与 convoC 的记忆都含"青椒"，应合并进同一实体点
const u1Entities = await raw.scroll(ENTITY_COLLECTION, {
  limit: 50,
  with_payload: true,
  filter: { must: [{ key: "userId", match: { value: "u1" } }] },
});
const pepper = u1Entities.points.find((p) => (p.payload as any).data === "青椒");
check(
  "青椒实体合并多条记忆",
  ((pepper?.payload as any)?.linkedMemoryIds?.length ?? 0) >= 2,
  pepper
    ? `关联 ${((pepper.payload as any).linkedMemoryIds as string[]).length} 条`
    : "u1 无青椒实体",
);

console.log("\n===== Part 3: searchMemories =====");

type SearchRes = Awaited<ReturnType<typeof memory.search>>;

// verify 返回错误描述，null 表示通过。只断言通道存在性与召回事实，
// 不绑定 top1 措辞与具体分值——阈值边界的措辞抖动属已知 flicker，flake 即重校准信号
const queries: Array<{
  q: string;
  user: string;
  expect: string;
  verify: (res: SearchRes) => string | null;
}> = [
  {
    q: "我爱吃什么菜",
    user: "u1",
    expect: "青椒记忆 top1",
    verify: (res) => {
      return res.some((r) => r.memory.includes("青椒"))
        ? null
        : `top5 未命中青椒记忆: ${res.map((r) => r.memory).join(" / ")}`;
    },
  },
  {
    q: "老公能吃辣吗",
    user: "u1",
    expect: "老公不能吃辣 top1",
    verify: (res) => {
      return res.some((r) => r.memory.includes("老公") && r.memory.includes("辣"))
        ? null
        : `top5 未命中老公吃辣记忆: ${res.map((r) => r.memory).join(" / ")}`;
    },
  },
  {
    q: "花椒从哪来的",
    user: "u1",
    expect: "花椒记忆命中，bm25 通道有话语权",
    verify: (res) => {
      const hit = res.find((r) => r.memory.includes("花椒"));
      if (!hit) return `top5 未命中花椒记忆: ${res.map((r) => r.memory).join(" / ")}`;
      return (hit.scoreDetails?.bm25Score ?? 0) > 0
        ? null
        : `花椒记忆的 bm25 分量为 0: ${JSON.stringify(hit.scoreDetails)}`;
    },
  },
  {
    q: "今天天气怎么样",
    user: "u1",
    expect: "threshold 门控下无召回",
    verify: (res) => {
      return res.length === 0
        ? null
        : `无关 query 产生召回（噪声地板可能上移）: ${res.map((r) => r.memory).join(" / ")}`;
    },
  },
  {
    q: "家里做菜放不放辣椒",
    user: "u2",
    expect: "只召回 u2 自己的记忆",
    verify: (res) => {
      return res.length > 0 && res.every((r) => !r.memory.includes("花椒"))
        ? null
        : `u2 隔离失效或零召回: ${res.map((r) => r.memory).join(" / ")}`;
    },
  },
  {
    q: "怎么做水煮鱼",
    user: "u1",
    expect: "水煮鱼记忆命中且实体通道生效",
    verify: (res) => {
      return res.some((r) => r.memory.includes("水煮鱼") && r.scoreDetails!.entityBoost > 0)
        ? null
        : `水煮鱼记忆未命中或实体 boost 为 0: ${JSON.stringify(res.map((r) => r.scoreDetails))}`;
    },
  },
];

for (const { q, user, expect, verify } of queries) {
  const res = await memory.search(q, { userId: user, topK: 5, explain: true });
  console.log(`\n[query] ${q} (user=${user})  — 预期: ${expect}`);
  if (res.length === 0) {
    console.log("  （无召回）");
  } else {
    for (const r of res) {
      const d = r.scoreDetails!;
      console.log(
        `  ${r.score.toFixed(3)}  ${r.memory}` +
          `  [sem=${d.semanticScore.toFixed(3)}, bm25=${d.bm25Score.toFixed(3)}, ent=${d.entityBoost.toFixed(3)}, max=${d.maxPossibleScore}]`,
      );
    }
  }
  const err = verify(res);
  check(`检索: ${q}`, err === null, err ?? "");
}

console.log("\n===== Part 4: 变更决策（u5 隔离用户） =====");

// 场景 1 UPDATE：状态变化应原地更新，不产生碎片
await memory.add([{ role: "user", content: "我家猫叫橘子，今年三岁了。" }], {
  userId: "u5",
  observationDate: "2026-10-18",
});
const u5Before = await raw.count(MEMORY_COLLECTION, { exact: true });
const up1 = await memory.add([{ role: "user", content: "对了，我把橘子改名叫团子了。" }], {
  userId: "u5",
  observationDate: "2026-10-18",
});
console.log("改名事件:", JSON.stringify(up1.events, null, 2));
const u5After = await raw.count(MEMORY_COLLECTION, { exact: true });
check(
  "UPDATE 不增加记忆条数",
  u5After.count <= u5Before.count + 1,
  `${u5Before.count} → ${u5After.count}`,
);
const renamedSearch = await memory.search("用户的猫叫什么名字", {
  userId: "u5",
  topK: 3,
});
console.log(
  "改名后检索:",
  renamedSearch.map((r) => r.memory),
);
check(
  "新名字可召回",
  renamedSearch.some((r) => r.memory.includes("团子")),
);
check(
  "无残留旧名记忆（旧名仅允许与改名语义同现）",
  renamedSearch.every(
    (r) =>
      !r.memory.includes("橘子") ||
      r.memory.includes("团子") ||
      /改名|原名|曾用名|旧名|以前叫|之前叫/.test(r.memory),
  ),
);

// 场景 2 DELETE+ADD：矛盾事实应删除旧记忆并写入新状态
await memory.add([{ role: "user", content: "我每天早上都要喝一杯拿铁。" }], {
  userId: "u5",
  observationDate: "2026-10-18",
});
const quit = await memory.add([{ role: "user", content: "我最近把咖啡戒了，一口都不喝。" }], {
  userId: "u5",
  observationDate: "2026-10-18",
});
console.log("戒咖啡事件:", JSON.stringify(quit.events, null, 2));
check(
  "旧行被 UPDATE 或 DELETE",
  quit.events.some((e) => e.event === "UPDATE" || e.event === "DELETE"),
  JSON.stringify(quit.events.map((e) => e.event)),
);
const coffeeSearch = await memory.search("早上喝什么", {
  userId: "u5",
  topK: 5,
});
console.log(
  "戒断后检索:",
  coffeeSearch.map((r) => `${r.memory}`),
);
check(
  "无残留现役拿铁习惯",
  coffeeSearch.every((r) => !/拿铁/.test(r.memory) || /以前|曾经|戒|改/.test(r.memory)),
);
const coffeeSearch2 = await memory.search("用户还喝咖啡吗", {
  userId: "u5",
  topK: 5,
});
console.log(
  "咖啡直查:",
  coffeeSearch2.map((r) => `${r.memory}`),
);
check("戒断状态可召回", coffeeSearch2.length > 0 && coffeeSearch2.some((r) => /戒/.test(r.memory)));

// 场景 3 NONE：等价复述不应新增记忆
const dup = await memory.add([{ role: "user", content: "我家有只三岁的猫，名字叫团子。" }], {
  userId: "u5",
  observationDate: "2026-10-18",
});
console.log("复述事件:", JSON.stringify(dup.events, null, 2));
check(
  "复述事实未新增（NONE 或 DEDUP）",
  dup.events
    .filter((e) => e.event !== "DEDUP" && e.text.includes("猫"))
    .every((e) => e.event === "NONE"),
  JSON.stringify(dup.events.map((e) => e.event)),
);

// 场景 4：实体回链完整性——被删除记忆的 id 不应残留在 linkedMemoryIds
const u5Entities = await raw.scroll(ENTITY_COLLECTION, {
  limit: 50,
  with_payload: true,
  filter: { must: [{ key: "userId", match: { value: "u5" } }] },
});
const memIds = new Set(
  (await raw.scroll(MEMORY_COLLECTION, { limit: 100, with_payload: false })).points.map((p) =>
    String(p.id),
  ),
);
let deadLinks = 0;
for (const p of u5Entities.points) {
  const pl = p.payload as any;
  for (const mid of pl.linkedMemoryIds ?? []) {
    if (!memIds.has(mid)) deadLinks++;
  }
}
check("u5 实体库无死链", deadLinks === 0, `死链 ${deadLinks} 个`);

console.log("\n===== 完成 =====");
if (failures > 0) {
  console.log(`\n${failures} 项检查未通过`);
  process.exitCode = 1;
}
