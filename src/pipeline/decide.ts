// 记忆变更决策（移植 mem0 getUpdateMemoryMessages 的 ADD/UPDATE/DELETE/NONE 语义）：
// 旧行与事实行拆成两组输出——比 mem0 的单数组更稳，LLM 全程不需要发明 id
// 矛盾场景输出「旧行 DELETE + 事实行 ADD」，新信息不丢失。
import { generateText, tool, type LanguageModel } from "ai";
import { z } from "zod";
import type { ExistingMemory, MemoryDecision } from "../types";

export const DECISION_SYSTEM_PROMPT = `# 角色

你是记忆库的变更管理器。输入两部分：旧记忆（带 id）与本次对话新抽取的事实。
逐行输出决策：事实行决定如何处置每条新事实，旧行仅在被推翻时标记删除。

# 事实行操作（facts，fact 原样返回）

- ADD：新信息，独立写入记忆库
- UPDATE：新事实修正了某条旧行记录的状态（改名、搬迁、换工作、戒断、数量变化、信息补充）
  → 给出 target_id（被修正的旧行 id）与 new_text（合并后的新文本，保留仍有价值的信息、丢弃过时信息）
- NONE：与某条旧行完全等价（两者读出的是同一个事实，仅措辞不同）。
  状态、名称、数量、归属发生变化时绝不算等价

# 旧行操作（oldMemories，id 原样返回）

- NONE：原样保留
- DELETE：这条记忆被新事实证明不成立，且新事实自会独立成条，没有值得并入它的信息

# 规则

- target_id 只能取输入旧记忆列表里的 id，fact 只能取输入事实列表里的原文，禁止编造
- new_text 必须自包含（脱离对话能读懂），保留信息量最大的版本
- 每条旧行、每条事实都必须输出恰好一条决策
- 输出前逐条自查：改变既有状态的事实，必须要么 UPDATE 某旧行，要么旧行 DELETE 且事实 ADD；
  事实与旧行"各说各话"的双 NONE 是错误——那意味着状态变化被静默丢弃

# 示例

- 旧记忆 "用户养了一只叫汤圆的猫" + 新事实 "用户把汤圆改名叫元宵了"：
  事实行 UPDATE，target_id 指向该旧行，new_text="用户养的猫改名叫元宵（原名汤圆）"
  ❌ 事实 NONE 且旧行 NONE——改名是状态变化，双双跳过会让旧名字继续被当成现役，新信息直接丢失
  ❌ 旧行 DELETE + 事实 ADD——纯名称修正应原地更新，不产生两条碎片
- 旧记忆 "用户每天午饭后睡半小时午觉" + 新事实 "用户新工作午休太短，已经不睡午觉了"：
  旧行 DELETE（午睡习惯已不成立），事实行 ADD "用户因午休太短已停止睡午觉"
- 旧记忆 "用户养了一只叫汤圆的猫" + 新事实 "汤圆特别怕生，家里来客人就躲床底"：
  旧行 NONE，事实行 ADD——同一实体的不同侧面各自成条
  ❌ 不要因为都提到汤圆就 UPDATE 合并
- 旧记忆 "用户家周末很少开火" + 新事实 "用户周末基本不做饭，都在外面吃"：
  旧行 NONE，事实行 NONE——同一事实的复述，两者读出的是同一个事实
  ❌ 不要因为措辞不同就 ADD`;

export interface MemoryDecider {
  decide(facts: string[], candidates: ExistingMemory[]): Promise<MemoryDecision>;
}

export function createMemoryDecider(model: LanguageModel): MemoryDecider {
  return {
    async decide(facts, candidates): Promise<MemoryDecision> {
      if (facts.length === 0) return { oldMemories: [], facts: [] };

      // 短 id 映射：让 LLM 逐字回显 36 位 UUID 不可靠，候选以 m1..mN 呈现，
      // 输出映射回真实 id
      const idToReal = new Map<string, string>();
      const labeled = candidates.map((c, i) => {
        const shortId = `m${i + 1}`;
        idToReal.set(shortId, c.id);
        return { id: shortId, text: c.text };
      });

      const sections: string[] = [];
      sections.push(
        `## 旧记忆\n${JSON.stringify(labeled.map((c) => ({ id: c.id, text: c.text })))}`,
      );
      sections.push(`## 新事实\n${JSON.stringify(facts)}`);

      let result: MemoryDecision = { oldMemories: [], facts: [] };

      const decideTool = tool({
        inputSchema: z.object({
          oldMemories: z.array(
            z.object({
              id: z.string(),
              event: z.enum(["NONE", "DELETE"]),
            }),
          ),
          facts: z.array(
            z.object({
              fact: z.string(),
              event: z.enum(["ADD", "NONE", "UPDATE"]),
              target_id: z.string().optional(),
              new_text: z.string().optional(),
            }),
          ),
        }),
        execute: (input) => {
          const toReal = (shortId?: string) =>
            shortId ? idToReal.get(shortId.trim().toLowerCase()) : undefined;
          result = {
            oldMemories: input.oldMemories.flatMap((o) => {
              const realId = toReal(o.id);
              return realId ? [{ id: realId, event: o.event }] : [];
            }),
            facts: input.facts.map((f) => ({
              fact: f.fact,
              event: f.event,
              ...(f.event === "UPDATE"
                ? {
                    targetId: toReal(f.target_id),
                    newText: f.new_text,
                  }
                : {}),
            })),
          };
        },
      });

      void (await generateText({
        model,
        instructions: DECISION_SYSTEM_PROMPT,
        messages: [{ role: "user", content: sections.join("\n\n") }],
        tools: {
          decide_memory: decideTool,
        },
        stopWhen: ({ steps }) => {
          if (steps.length) {
            const lastStep = steps[steps.length - 1];
            return lastStep.toolCalls.some((call) => call.toolName === "decide_memory");
          }
          return false;
        },
      }));

      return result;
    },
  };
}
