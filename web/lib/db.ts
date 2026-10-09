import "server-only";

import { Pool, types } from "pg";

// DATE columns as "YYYY-MM-DD" strings: a JS Date would shift them by the
// server's timezone offset.
types.setTypeParser(1082, (value: string) => value);

const globalForPool = globalThis as unknown as { ecoPool?: Pool };

/** One pool per server process (reused across dev hot reloads). */
export function db(): Pool {
  if (!globalForPool.ecoPool) {
    const connectionString = process.env.DATABASE_URL;
    if (!connectionString) {
      throw new Error(
        "DATABASE_URL is not set (copy .env.local.example to .env.local)",
      );
    }
    globalForPool.ecoPool = new Pool({ connectionString, max: 4 });
  }
  return globalForPool.ecoPool;
}
