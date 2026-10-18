import { beforeEach, describe, expect, it, vi } from "vitest";

// execute 的入参在测试里逐用例改写；generateText 捕获传入的 tools 并直接调 execute，
// 钉死短 id 映射与蛇形→驼峰转换（踩坑 #14：typecheck 抓不到的静默丢字段类 bug）
const toolOutput = vi.hoisted(() => ({
  oldMemories: [] as Array<{ id: string; event: "NONE" | "DELETE" }>,
  facts: [] as Array<{
    fact: string;
    event: "ADD" | "NONE" | "UPDATE";
    target_id?: string;
    new_text?: string;
  }>,
}));

const generateTextMock = vi.hoisted(() => vi.fn());

vi.mock("ai", () => ({
  tool: (spec: unknown) => spec,
  generateText: generateTextMock,
}));

import { createMemoryDecider } from "./decide";
import type { ExistingMemory } from "../types";

generateTextMock.mockImplementation(async ({ tools }: any) => {
  await (tools as any).decide_memory.execute(toolOutput);
  return { steps: [{ toolCalls: [{ toolName: "decide_memory" }] }] };
});

const decider = createMemoryDecider({} as any);

const candidates: ExistingMemory[] = [
  { id: "uuid-aaa", text: "用户养了一只叫橘子的猫" },
  { id: "uuid-bbb", text: "用户每天早上喝一杯拿铁" },
];

beforeEach(() => {
  toolOutput.oldMemories = [];
  toolOutput.facts = [];
  generateTextMock.mockClear();
});

describe("createMemoryDecider 决策映射", () => {
  it("facts 为空时短路，不调用 LLM", async () => {
    const d = await decider.decide([], candidates);
    expect(d).toEqual({ oldMemories: [], facts: [] });
    expect(generateTextMock).not.toHaveBeenCalled();
  });

  it("旧行短 id 映射回真实 UUID", async () => {
    toolOutput.oldMemories = [{ id: "m2", event: "DELETE" }];
    toolOutput.facts = [{ fact: "用户已戒咖啡", event: "ADD" }];
    const d = await decider.decide(["用户已戒咖啡"], candidates);
    expect(d.oldMemories).toEqual([{ id: "uuid-bbb", event: "DELETE" }]);
    expect(d.facts).toEqual([{ fact: "用户已戒咖啡", event: "ADD" }]);
  });

  it("UPDATE 的蛇形字段映射为驼峰且目标 id 还原", async () => {
    toolOutput.facts = [
      {
        fact: "用户把橘子改名叫团子了",
        event: "UPDATE",
        target_id: "m1",
        new_text: "用户养的猫改名叫团子（原名橘子）",
      },
    ];
    const d = await decider.decide(["用户把橘子改名叫团子了"], candidates);
    expect(d.facts[0]).toEqual({
      fact: "用户把橘子改名叫团子了",
      event: "UPDATE",
      targetId: "uuid-aaa",
      newText: "用户养的猫改名叫团子（原名橘子）",
    });
  });

  it("非 UPDATE 事实不携带 targetId/newText 键", async () => {
    toolOutput.facts = [
      { fact: "用户喜欢青椒", event: "ADD" },
      { fact: "用户喜欢青椒", event: "NONE" },
    ];
    const d = await decider.decide(["用户喜欢青椒"], candidates);
    expect("targetId" in d.facts[0]).toBe(false);
    expect("newText" in d.facts[0]).toBe(false);
    expect("targetId" in d.facts[1]).toBe(false);
  });

  it("LLM 编造的短 id 旧行被丢弃", async () => {
    toolOutput.oldMemories = [
      { id: "m99", event: "DELETE" },
      { id: "m1", event: "NONE" },
    ];
    const d = await decider.decide(["某事实"], candidates);
    expect(d.oldMemories).toEqual([{ id: "uuid-aaa", event: "NONE" }]);
  });

  it("短 id 容忍空白与大小写", async () => {
    toolOutput.oldMemories = [{ id: " M1 ", event: "DELETE" }];
    const d = await decider.decide(["某事实"], candidates);
    expect(d.oldMemories).toEqual([{ id: "uuid-aaa", event: "DELETE" }]);
  });

  it("UPDATE 指向编造 id 时 targetId 为 undefined（由管线退化为 ADD）", async () => {
    toolOutput.facts = [{ fact: "f", event: "UPDATE", target_id: "m42", new_text: "t" }];
    const d = await decider.decide(["f"], candidates);
    expect(d.facts[0].targetId).toBeUndefined();
    expect(d.facts[0].newText).toBe("t");
  });
});
