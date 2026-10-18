import { Jieba } from "@node-rs/jieba";
import { dict } from "@node-rs/jieba/dict";

const jieba = Jieba.withDict(dict);

export function tokenizeForCandidates(text: string): string[] {
  return (
    jieba
      .cut(text.toLowerCase())
      .map((t) => t.trim())
      // 保留含 Unicode 字母或数字的 token；\w 为 ASCII 语义，不适用于中文
      .filter((t) => /[\p{L}\p{N}]/u.test(t))
  );
}
