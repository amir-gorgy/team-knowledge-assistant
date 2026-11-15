// Typed client for the API's GET /health.
//
//   node health-client.ts [baseUrl]     run (Node strips the types, no build step)
//   npx tsc --noEmit                    type-check

// The two bodies /health can send. `status` is the discriminant: checking it
// tells TypeScript which of the two shapes you have (a "discriminated union").
type HealthOk = {
  status: "ok";
  db: "ok";
  pgvector: string | null; // extension version, null if the extension is missing
};

type HealthDegraded = {
  status: "degraded";
  db: "unreachable";
};

export type Health = HealthOk | HealthDegraded;

// res.json() returns data TypeScript knows nothing about, so check it before
// trusting it. The `x is Health` return type is a type predicate: where this
// returns true, TypeScript treats x as Health.
export function isHealth(x: unknown): x is Health {
  if (typeof x !== "object" || x === null || !("status" in x)) return false;
  if (x.status === "ok") {
    return (
      "db" in x && x.db === "ok" &&
      "pgvector" in x && (typeof x.pgvector === "string" || x.pgvector === null)
    );
  }
  if (x.status === "degraded") {
    return "db" in x && x.db === "unreachable";
  }
  return false;
}

export async function getHealth(baseUrl = "http://localhost:8000"): Promise<Health> {
  // Throws a TypeError if the API isn't reachable at all; the caller handles that.
  const res = await fetch(`${baseUrl}/health`, { method: "GET", signal: AbortSignal.timeout(3000) });
  // Don't throw on every non-2xx: a 503 still carries a degraded body.
  const body: unknown = await res.json();
  if (!isHealth(body)) {
    throw new Error(`Unexpected /health response (HTTP ${res.status}): ${JSON.stringify(body)}`);
  }
  return body;
}

export function describe(h: Health): string {
  // Before this check, h.pgvector is a type error: HealthDegraded has no pgvector.
  switch (h.status) {
    case "ok":
      return `OK, pgvector ${h.pgvector ?? "not installed"}`;
    case "degraded":
      return "Degraded: database unreachable";
    default: {
      // If a third status is added to Health and not handled above, this line stops compiling.
      const unhandled: never = h;
      return unhandled;
    }
  }
}

async function main(): Promise<void> {
  const baseUrl = process.argv[2] ?? "http://localhost:8000";
  try {
    const health = await getHealth(baseUrl);
    console.log(describe(health));
    process.exitCode = health.status === "ok" ? 0 : 1;
  } catch (err) {
    console.log(`API not reachable at ${baseUrl}: ${reason(err)}`);
    process.exitCode = 2;
  }
}

// fetch wraps network failures: the real reason is on err.cause, and a refused
// connection has an empty message but a code like "ECONNREFUSED".
function reason(err: unknown): string {
  const cause = err instanceof Error && err.cause !== undefined ? err.cause : err;
  if (typeof cause === "object" && cause !== null && "code" in cause && typeof cause.code === "string") {
    return cause.code;
  }
  return cause instanceof Error ? cause.message : String(cause);
}

await main();
