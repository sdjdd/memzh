// 评测卫生：提示词示例与评测集/回归场景零词汇重叠。
// 提示词里印着正确答案的措辞，会让对应评测项退化为同义反复——
// 模型可能是在回显示例而非运用规则（踩坑 #15）。本测试把"重叠"变成 CI 可见的失败：
// 任何新示例入 prompt 前，先确认它用词不在被评测词汇表里。
import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { EXTRACTION_SYSTEM_PROMPT } from "../src/extraction/prompt";
import { DECISION_SYSTEM_PROMPT } from "../src/pipeline/decide";

// tests/e2e.ts 的场景对话无法静态 import（脚本非导出模块），人工维护显著名词清单
const E2E_NOUNS = ["青椒", "花椒", "汉源", "水煮鱼", "拿铁", "咖啡", "团子", "成都", "虎皮"];

const corpus: { conversations: { expectEntities?: string[] }[] } = JSON.parse(
  readFileSync(new URL("./corpus.json", import.meta.url), "utf8"),
);
const queries: { queries: { expect: string[] }[] } = JSON.parse(
  readFileSync(new URL("./queries.json", import.meta.url), "utf8"),
);

// 被评测词汇表 = 语料对标实体 ∪ 查询判定关键词 ∪ e2e 场景名词。
// 判定靠这些词给分，它们出现在提示词里即为泄漏
const scoredVocab = new Set<string>();
for (const c of corpus.conversations) for (const e of c.expectEntities ?? []) scoredVocab.add(e);
for (const q of queries.queries) for (const e of q.expect) if (e) scoredVocab.add(e);
for (const e of E2E_NOUNS) scoredVocab.add(e);

const prompts = EXTRACTION_SYSTEM_PROMPT + DECISION_SYSTEM_PROMPT;

describe("评测卫生：提示词与评测词汇零重叠", () => {
  it.each([...scoredVocab])("提示词不得出现被评测词汇「%s」", (word) => {
    expect(prompts).not.toContain(word);
  });
});
