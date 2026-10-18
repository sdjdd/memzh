// 采用 tool-call 而非 response_format: json_object，LLM 对工具调用参数的结构遵循度更好。
import { generateText, tool, type LanguageModel } from "ai";
import { z } from "zod";
import type { ChatMessage, ExtractedMemory, ExtractOptions } from "../types";
import { warnOnce } from "../utils";
import { EXTRACTION_SYSTEM_PROMPT } from "./prompt";

export type { ExtractOptions };

const TRUNCATE = 300;

function formatHistory(messages?: ChatMessage[]): string {
  if (!messages?.length) return "";
  return messages.map((m) => `${m.role}: ${m.content.slice(0, TRUNCATE)}`).join("\n");
}

function buildUserPrompt(messages: ChatMessage[], opts: ExtractOptions): string {
  const observationDate = opts.observationDate ?? new Date().toISOString().slice(0, 10);
  const sections: string[] = [];

  sections.push(
    `## 已有记忆\n${JSON.stringify(opts.existingMemories ?? [])}\n（仅用于查重和关联，不要从中抽取新内容）`,
  );
  sections.push(`## 之前的消息\n${formatHistory(opts.lastKMessages) || "（无）"}`);
  sections.push(`## 新消息\n${JSON.stringify(messages)}`);
  sections.push(`## 观测日期\n${observationDate}`);
  if (opts.customInstructions) {
    sections.push(`## 自定义指令\n${opts.customInstructions}`);
  }
  return sections.join("\n\n");
}

export interface MemoryExtractor {
  extract(messages: ChatMessage[], options?: ExtractOptions): Promise<ExtractedMemory[]>;
}

/** 空抽取重试门槛（用户侧内容总字符数）：低于此值的空抽取视为合法输出（寒暄类对话） */
const RETRY_MIN_USER_CHARS = 10;

export function createExtractor(model: LanguageModel): MemoryExtractor {
  async function extractOnce(
    messages: ChatMessage[],
    options: ExtractOptions,
  ): Promise<ExtractedMemory[]> {
    const userPrompt = buildUserPrompt(messages, options);

    let result: ExtractedMemory[] = [];

    const memTool = tool({
      inputSchema: z.object({
        memory: z.array(
          z.object({
            id: z.string(),
            text: z.string(),
            attributed_to: z.enum(["user", "assistant"]),
            entities: z.array(z.string()).optional(),
          }),
        ),
      }),
      execute: (input) => {
        result = input.memory
          .filter((m) => m.text.trim())
          .map((m) => ({
            text: m.text.trim(),
            attributedTo: m.attributed_to,
            ...(m.entities?.length ? { entities: m.entities } : {}),
          }));
      },
    });

    void (await generateText({
      model,
      instructions: EXTRACTION_SYSTEM_PROMPT,
      messages: [{ role: "user", content: userPrompt }],
      tools: {
        add_memory: memTool,
      },
      stopWhen: ({ steps }) => {
        if (steps.length) {
          const lastStep = steps[steps.length - 1];
          return lastStep.toolCalls.some((call) => call.toolName === "add_memory");
        }
        return false;
      },
    }));

    return result;
  }

  return {
    async extract(messages, options = {}) {
      const result = await extractOnce(messages, options);
      if (result.length > 0) return result;
      // LLM 抽取偶发对富内容对话返回空结果；利用非确定性重试一次。
      // 仅当用户侧内容达到可抽取量级时才重试，寒暄对话的空抽取是合法输出
      const userChars = messages
        .filter((m) => m.role === "user")
        .reduce((n, m) => n + m.content.length, 0);
      if (userChars < RETRY_MIN_USER_CHARS) return result;
      const retried = await extractOnce(messages, options);
      if (retried.length === 0) {
        warnOnce("extraction:empty-after-retry", "extraction: 富内容对话抽取重试后仍为空");
      }
      return retried;
    },
  };
}
