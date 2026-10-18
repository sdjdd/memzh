// .env 加载：封装 Node 原生 process.loadEnvFile（Node ≥20.12），不自行解析。
// 保留薄壳的原因：路径按模块定位，从任意工作目录运行均指向项目根的 .env
// （原生无参调用按工作目录查找）；文件缺失时静默跳过（原生会抛出异常）。
// 优先级语义与原生一致：已存在的真实环境变量不被 .env 覆盖。
import { fileURLToPath } from "node:url";

export function loadEnv(): void {
  try {
    process.loadEnvFile(fileURLToPath(new URL("../.env", import.meta.url)));
  } catch {
    // 缺少 .env 时依赖真实环境变量运行
  }
}
